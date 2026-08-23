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
// Weight storage
// --------------
// wmem[n][k] holds the 9-tap kernel for output channel n, input channel k, and
// is read combinationally on the window stream's m_ch. That infers LUTRAM:
// 9*16 = 144 bits x CH_IN deep per MAC. At CH_IN=64 that is ~144 LUTs per MAC,
// ~3.5k LUTs at N_OUT=24, which fits comfortably in 53k.
//
// At CH_IN=128 this doubles and BRAM becomes preferable -- but BRAM reads are
// registered, so the window would need a matching delay stage. Do that only when
// the LUT budget actually demands it, and re-run the integration testbench when
// you do: adding a pipeline stage on one of two converging paths is exactly the
// change that breaks composition while every unit test still passes.
//
// Weights load through a simple write port. DMA loading (PROJECT_CONTEXT.md
// section 11 step 4, replacing 186k MMIO writes) wraps this later.
//
module conv_layer #(
    parameter int DATA_W    = 16,
    parameter int WEIGHT_W  = 16,
    parameter int IMG_W     = 112,
    parameter int IMG_H     = 112,
    parameter int CH_IN     = 6,
    parameter int N_OUT     = 4,
    parameter int ACC_W     = 48,
    parameter int OUT_SHIFT = WEIGHT_W - 1
)(
    input  logic                        clk,
    input  logic                        rst_n,

    // weight load port
    input  logic                        w_we,
    input  logic [$clog2(N_OUT > 1 ? N_OUT : 2)-1:0]  w_out_ch,
    input  logic [$clog2(CH_IN > 1 ? CH_IN : 2)-1:0]  w_in_ch,
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

    localparam int CHW = (CH_IN <= 1) ? 1 : $clog2(CH_IN);

    logic [9*DATA_W-1:0] win_data;
    logic [CHW-1:0]      win_ch;
    logic                win_first, win_last_ch, win_last_frame;
    logic                win_valid, win_ready;

    line_buffer_mc #(
        .DATA_W (DATA_W), .IMG_W (IMG_W), .IMG_H (IMG_H), .CHANNELS (CH_IN)
    ) u_lb (
        .clk (clk), .rst_n (rst_n),
        .s_data (s_data), .s_valid (s_valid), .s_ready (s_ready),
        .m_window (win_data), .m_ch (win_ch),
        .m_first (win_first), .m_last_ch (win_last_ch), .m_last_frame (win_last_frame),
        .m_valid (win_valid), .m_ready (win_ready)
    );

    // Kernels, indexed [output channel][input channel].
    logic [9*WEIGHT_W-1:0] wmem [N_OUT][CH_IN];

    always_ff @(posedge clk) begin
        if (w_we) wmem[w_out_ch][w_in_ch] <= w_data;
    end

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
            conv_mac_3x3 #(
                .DATA_W (DATA_W), .WEIGHT_W (WEIGHT_W),
                .ACC_W (ACC_W), .OUT_SHIFT (OUT_SHIFT)
            ) u_mac (
                .clk (clk), .rst_n (rst_n),
                .s_window  (win_data),
                .s_weights (wmem[n][win_ch]),
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
