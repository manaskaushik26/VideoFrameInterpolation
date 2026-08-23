`timescale 1ns / 1ps
//
// conv_layer_mc
//
// A complete multi-channel 3x3 convolution producing one output channel:
// CHANNELS input channels in, one convolved channel out. This is line_buffer_mc
// feeding conv_mac_3x3, with the kernel selected per channel.
//
//     out(y,x) = sum over k,r,c of  w(k,r,c) * in(k, y+r-1, x+c-1)
//
// Cross-correlation, unflipped, matching nn.Conv2d -- see conv_layer_1ch.
//
// Weight layout: CHANNELS kernels of 9 taps, input channel major.
//
//     weights[(k*9 + r*3 + c) * WEIGHT_W +: WEIGHT_W]
//
// Scaling to a real layer
// -----------------------
// A layer with N output channels is N instances of conv_mac_3x3 sharing one
// line_buffer_mc, each with its own kernel set, all consuming the same window
// stream. The line buffer is the expensive part (BRAM) and is amortised across
// all N; the MACs are the DSP cost, 9 each. At 220 DSPs that is ~24 output
// channels in flight. This module is the N=1 case, and the point at which the
// window stream and the MAC's first/last protocol are proven to line up.
//
module conv_layer_mc #(
    parameter int DATA_W    = 16,
    parameter int WEIGHT_W  = 16,
    parameter int IMG_W     = 112,
    parameter int IMG_H     = 112,
    parameter int CHANNELS  = 6,
    parameter int ACC_W     = 48,
    parameter int OUT_SHIFT = WEIGHT_W - 1
)(
    input  logic                            clk,
    input  logic                            rst_n,

    input  logic [CHANNELS*9*WEIGHT_W-1:0]  weights,

    input  logic [DATA_W-1:0]               s_data,
    input  logic                            s_valid,
    output logic                            s_ready,

    output logic signed [DATA_W-1:0]        m_data,
    output logic                            m_sat,
    output logic                            m_valid,
    input  logic                            m_ready,
    output logic                            m_last
);

    localparam int CHW = $clog2(CHANNELS > 1 ? CHANNELS : 2);

    logic [9*DATA_W-1:0] win_data;
    logic [CHW-1:0]      win_ch;
    logic                win_first, win_last_ch, win_last_frame;
    logic                win_valid, win_ready;

    line_buffer_mc #(
        .DATA_W (DATA_W), .IMG_W (IMG_W), .IMG_H (IMG_H), .MAX_CH (CHANNELS)
    ) u_lb (
        .clk (clk), .rst_n (rst_n),
        .cfg_channels ($clog2(CHANNELS+1)'(CHANNELS)),
        .s_data (s_data), .s_valid (s_valid), .s_ready (s_ready),
        .m_window (win_data), .m_ch (win_ch),
        .m_first (win_first), .m_last_ch (win_last_ch), .m_last_frame (win_last_frame),
        .m_valid (win_valid), .m_ready (win_ready),
        // This wrapper selects its kernel combinationally from a flat port, so
        // it has no per-channel memory to read early.
        .m_rd_ch (), .m_rd_en ()
    );

    // Kernel for the channel currently on the window stream.
    logic [9*WEIGHT_W-1:0] kernel;
    assign kernel = weights[win_ch * (9*WEIGHT_W) +: 9*WEIGHT_W];

    conv_mac_3x3 #(
        .DATA_W (DATA_W), .WEIGHT_W (WEIGHT_W), .ACC_W (ACC_W)
    ) u_mac (
        .clk (clk), .rst_n (rst_n),
        .cfg_shift (6'(OUT_SHIFT)),
        .s_window (win_data), .s_weights (kernel),
        .s_first (win_first), .s_last (win_last_ch),
        .s_tag   (win_last_frame),
        .s_valid (win_valid), .s_ready (win_ready),
        .m_data (m_data), .m_sat (m_sat), .m_tag (m_last),
        .m_valid (m_valid), .m_ready (m_ready)
    );

endmodule
