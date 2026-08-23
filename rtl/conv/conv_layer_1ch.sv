`timescale 1ns / 1ps
//
// conv_layer_1ch
//
// A complete single-channel 3x3 convolution: pixel stream in, convolved pixel
// stream out. This is line_buffer_3x3 feeding conv_mac_3x3, and it exists as
// much to prove those two compose as to be useful on its own -- the superseded
// design's modules each passed a unit testbench and the integrated result
// computed a constant zero.
//
// Cross-correlation, not true convolution
// ---------------------------------------
// The kernel is applied without flipping:
//
//     out(y,x) = sum over r,c of  w(r,c) * in(y+r-1, x+c-1)
//
// which matches the window layout of line_buffer_3x3 tap for tap, and matches
// what PyTorch's nn.Conv2d actually computes. The trained weights can therefore
// be loaded in their natural order. Do not add a flip later "to make it a real
// convolution" -- that would silently mirror every kernel.
//
// Multi-channel layers replace this wrapper rather than wrapping it: they need
// C parallel line buffers and a first/last channel sequence into the MAC. Here
// s_first and s_last are both tied high, so every window produces one output.
//
module conv_layer_1ch #(
    parameter int DATA_W    = 16,
    parameter int WEIGHT_W  = 16,
    parameter int IMG_W     = 112,
    parameter int IMG_H     = 112,
    parameter int ACC_W     = 48,
    parameter int OUT_SHIFT = WEIGHT_W - 1
)(
    input  logic                        clk,
    input  logic                        rst_n,

    // kernel, held stable for the frame; same flattened layout as the window
    input  logic [9*WEIGHT_W-1:0]       weights,

    input  logic [DATA_W-1:0]           s_data,
    input  logic                        s_valid,
    output logic                        s_ready,

    output logic signed [DATA_W-1:0]    m_data,
    output logic                        m_sat,
    output logic                        m_valid,
    input  logic                        m_ready,
    output logic                        m_last
);

    logic [9*DATA_W-1:0] win_data;
    logic                win_valid, win_ready, win_last;

    line_buffer_3x3 #(
        .DATA_W (DATA_W), .IMG_W (IMG_W), .IMG_H (IMG_H)
    ) u_lb (
        .clk (clk), .rst_n (rst_n),
        .s_data (s_data), .s_valid (s_valid), .s_ready (s_ready),
        .m_window (win_data), .m_valid (win_valid), .m_ready (win_ready), .m_last (win_last)
    );

    conv_mac_3x3 #(
        .DATA_W (DATA_W), .WEIGHT_W (WEIGHT_W), .ACC_W (ACC_W), .OUT_SHIFT (OUT_SHIFT)
    ) u_mac (
        .clk (clk), .rst_n (rst_n),
        .s_window (win_data), .s_weights (weights),
        .s_first (1'b1), .s_last (1'b1),          // one channel: every beat completes a pixel
        .s_valid (win_valid), .s_ready (win_ready),
        .m_data (m_data), .m_sat (m_sat),
        .m_valid (m_valid), .m_ready (m_ready)
    );

    // Carry end-of-frame alongside the MAC's output register. Valid only because
    // s_last is tied high, so each accepted window yields exactly one output on
    // the next cycle and this flag stays in step with m_data. A multi-channel
    // version cannot reuse this -- there the relationship is C beats to one
    // output, and end-of-frame has to be tracked on the last channel only.
    always_ff @(posedge clk) begin
        if (!rst_n)                        m_last <= 1'b0;
        else if (win_valid && win_ready)   m_last <= win_last;
    end

endmodule
