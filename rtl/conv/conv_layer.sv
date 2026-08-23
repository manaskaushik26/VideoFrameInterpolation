`timescale 1ns / 1ps
//
// conv_layer
//
// A 3x3 convolution layer producing N_OUT output channels in parallel from
// CH_IN input channels:
//
//     out(n, y, x) = sum over k,r,c of  w(n, k, r, c) * in(k, y+r-1, x+c-1)
//
// One line_buffer_mc feeds N_OUT conv_mac_3x3 instances. The line buffer emits
// one window per input channel; every MAC sees the same window and applies its
// own kernel, so after CH_IN beats all N_OUT output pixels complete together.
//
// This is the shape that makes the compute budget work. The line buffer is the
// BRAM cost and is paid once regardless of N_OUT; each MAC is 9 DSP48E1. With
// 220 DSPs the ceiling is N_OUT = 24.
//
//     conv3 at 112x112, 64->128:  112*112*128*64*9 = 924.8 MMAC
//     at 9 MAC/cycle/instance and N_OUT=24, ~4.3M cycles = 43 ms at 100 MHz
//
// A layer with more output channels than N_OUT is run as ceil(C_out/N_OUT)
// passes over the input. Where that input lives between passes -- and between
// layers, since a 112x112x64 map is 12.8 Mbit and cannot stay on chip -- is the
// sequencer's problem, not this module's.
//
// Weight storage: one simple dual-port BRAM per MAC
// -------------------------------------------------
// Each MAC has its own kernel memory, MAX_CH deep and 9*WEIGHT_W wide, written
// by the loader and read on the window stream's channel index.
//
// These must be BRAM. An earlier version declared them as arrays read
// combinationally, expecting LUTRAM; Vivado inferred flip-flops instead, and at
// MAX_CH=128, N_OUT=24 that measured 138,248 LUT (260% of the device) and
// 457,663 registers (430%) -- 24 x 128 x 9 x 16b is 442 Kbit of kernels. As
// BRAM the same storage is ~48 BRAM36 against 140 available.
//
// The registered read costs no latency and needs no alignment stage, because
// the read is issued from line_buffer_mc's *stage-A* channel index (m_rd_ch,
// enabled by m_rd_en) rather than the window's own m_ch. Stage A runs exactly
// one cycle ahead of the window, so the BRAM output lands in the same cycle as
// the window it belongs to. When the pipeline stalls, m_rd_en is low and the
// BRAM holds -- which is correct, because stage B is holding too.
//
// Do not "simplify" this to read on m_ch. That would be a cycle late, and every
// output pixel would be convolved with the previous channel's kernel.
//
// Weights load through a simple write port. DMA loading (PROJECT_CONTEXT.md
// section 11 step 4, replacing 186k MMIO writes) wraps this later.
//
module conv_layer #(
    parameter int DATA_W    = 16,
    parameter int WEIGHT_W  = 16,
    parameter int IMG_W     = 112,
    parameter int IMG_H     = 112,
    parameter int MAX_CH    = 8,   // sizes the line buffer and weight memory
    parameter int N_OUT     = 4,   // physical MAC instances; 24 max on 220 DSPs
    parameter int ACC_W     = 48,
    parameter int SHIFT_W   = 6
)(
    input  logic                        clk,
    input  logic                        rst_n,

    // Runtime layer configuration. Both must be stable while a frame is in
    // flight. See line_buffer_mc and conv_mac_3x3 for why these are not
    // parameters: one bitstream has to run all six layers.
    input  logic [$clog2(MAX_CH+1)-1:0] cfg_channels,
    input  logic [SHIFT_W-1:0]          cfg_shift,

    // weight load port
    input  logic                        w_we,
    input  logic [$clog2(N_OUT > 1 ? N_OUT : 2)-1:0]  w_out_ch,
    input  logic [$clog2(MAX_CH > 1 ? MAX_CH : 2)-1:0] w_in_ch,
    input  logic [9*WEIGHT_W-1:0]       w_data,

    // pixel stream in: raster order, channel-minor
    input  logic [DATA_W-1:0]           s_data,
    input  logic                        s_valid,
    output logic                        s_ready,

    // N_OUT output channels, all valid on the same beat
    output logic [N_OUT*DATA_W-1:0]     m_data,
    output logic [N_OUT-1:0]            m_sat,
    output logic                        m_valid,
    input  logic                        m_ready,
    output logic                        m_last
);

    localparam int CHW = $clog2(MAX_CH > 1 ? MAX_CH : 2);

    logic [9*DATA_W-1:0] win_data;
    logic [CHW-1:0]      win_rd_ch;
    logic                win_rd_en;
    logic                win_first, win_last_ch, win_last_frame;
    logic                win_valid, win_ready;

    line_buffer_mc #(
        .DATA_W (DATA_W), .IMG_W (IMG_W), .IMG_H (IMG_H), .MAX_CH (MAX_CH)
    ) u_lb (
        .clk (clk), .rst_n (rst_n),
        .cfg_channels (cfg_channels),
        .s_data (s_data), .s_valid (s_valid), .s_ready (s_ready),
        // m_ch is deliberately unused: the weight BRAMs are addressed from the
        // early m_rd_ch instead, which is a cycle ahead and therefore aligned.
        .m_window (win_data), .m_ch (),
        .m_first (win_first), .m_last_ch (win_last_ch), .m_last_frame (win_last_frame),
        .m_valid (win_valid), .m_ready (win_ready),
        .m_rd_ch (win_rd_ch), .m_rd_en (win_rd_en)
    );

    // Every MAC is driven identically apart from its kernel, so their handshakes
    // move in lockstep. Rather than rely on that and read instance 0, the status
    // lines are reduced across all instances below: identical inputs make the
    // reduction equivalent, it costs one gate, and if they ever did diverge the
    // AND is the safe answer instead of a silent corruption. The testbench also
    // asserts they agree.
    logic [N_OUT-1:0] mac_s_ready, mac_m_valid, mac_m_tag;

    genvar n;
    generate
        for (n = 0; n < N_OUT; n++) begin : g_mac
            // One SDP BRAM per MAC: written by the loader, read a cycle early
            // from the line buffer's stage-A index so the output aligns with the
            // window with no delay stage.
            logic [9*WEIGHT_W-1:0] wmem [MAX_CH];
            logic [9*WEIGHT_W-1:0] wrd;

            always_ff @(posedge clk) begin
                if (w_we && (int'(w_out_ch) == n)) wmem[w_in_ch] <= w_data;
                if (win_rd_en)                     wrd <= wmem[win_rd_ch];
            end

            conv_mac_3x3 #(
                .DATA_W (DATA_W), .WEIGHT_W (WEIGHT_W),
                .ACC_W (ACC_W), .SHIFT_W (SHIFT_W)
            ) u_mac (
                .clk (clk), .rst_n (rst_n),
                .cfg_shift (cfg_shift),
                .s_window  (win_data),
                .s_weights (wrd),
                .s_first   (win_first),
                .s_last    (win_last_ch),
                .s_tag     (win_last_frame),
                .s_valid   (win_valid),
                .s_ready   (mac_s_ready[n]),
                .m_data    (m_data[n*DATA_W +: DATA_W]),
                .m_sat     (m_sat[n]),
                .m_tag     (mac_m_tag[n]),
                .m_valid   (mac_m_valid[n]),
                .m_ready   (m_ready)
            );
        end
    endgenerate

    assign win_ready = &mac_s_ready;
    assign m_valid   = &mac_m_valid;
    assign m_last    = &mac_m_tag;

endmodule
