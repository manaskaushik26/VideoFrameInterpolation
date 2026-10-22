`timescale 1ns / 1ps
//
// conv_stage -- one complete network layer: serial pixel stream in, serial
// pixel stream out.
//
//     conv_layer  ->  act_unit  ->  axis_serializer
//
// The input is (row, col, channel) channel-minor with CH_IN channels; the output
// is the same shape with N_OUT channels. Input and output formats are therefore
// identical, which is the whole point: conv_stage instances chain directly into
// one another with no glue, and the same shape is what the AXI-Stream DMA path
// carries to and from DDR.
//
// That chaining is only valid while N_OUT is the layer's *full* output channel
// count. Once a layer needs more output channels than the DSP budget allows in
// one pass (N_OUT = 24 max, measured), it becomes ceil(C_out/N_OUT) passes and
// the partial results have to be concatenated somewhere off-chip -- a
// 112x112x64 map is 12.8 Mbit and cannot stay on-chip. That staging is the
// sequencer's job and is still an open design question.
//
module conv_stage #(
    parameter int DATA_W    = 16,
    parameter int WEIGHT_W  = 16,
    parameter int IMG_W     = 112,
    parameter int IMG_H     = 112,
    parameter int MAX_CH    = 8,
    parameter int N_OUT     = 4,
    parameter int ACC_W     = 48,
    parameter int SHIFT_W   = 6,
    parameter int ACT_MODE  = 1        // 0 linear, 1 ReLU
)(
    input  logic                        clk,
    input  logic                        rst_n,

    input  logic [$clog2(MAX_CH+1)-1:0] cfg_channels,
    input  logic [SHIFT_W-1:0]          cfg_shift,
    input  logic                        cfg_linear,     // 1: bypass ReLU (flow layer)

    input  logic                        w_we,
    input  logic [$clog2(N_OUT > 1 ? N_OUT : 2)-1:0]  w_out_ch,
    input  logic [$clog2(MAX_CH > 1 ? MAX_CH : 2)-1:0] w_in_ch,
    input  logic [9*WEIGHT_W-1:0]       w_data,

    input  logic [DATA_W-1:0]           s_data,
    input  logic                        s_valid,
    output logic                        s_ready,

    output logic [DATA_W-1:0]           m_data,
    output logic                        m_last,
    output logic                        m_valid,
    input  logic                        m_ready,

    // Sticky: set if any output of this layer clipped since the last clear.
    // This is the diagnostic for OUT_SHIFT being wrong for the layer -- a shift
    // too small clips and destroys the activation, a shift too large throws away
    // precision. The per-layer shifts are a property of the trained model and are
    // still unknown, so software will need this to tune them: clear before a
    // frame, read after. Dropping it would mean a layer could silently clip every
    // pixel and look like it was working.
    input  logic                        sat_clear,
    output logic                        sat_sticky
);

    logic [N_OUT*DATA_W-1:0] conv_data, act_data;
    logic [N_OUT-1:0]        conv_sat;
    logic                    conv_valid, conv_ready, conv_last;

    conv_layer #(
        .DATA_W (DATA_W), .WEIGHT_W (WEIGHT_W),
        .IMG_W (IMG_W), .IMG_H (IMG_H),
        .MAX_CH (MAX_CH), .N_OUT (N_OUT),
        .ACC_W (ACC_W), .SHIFT_W (SHIFT_W)
    ) u_conv (
        .clk (clk), .rst_n (rst_n),
        .cfg_channels (cfg_channels), .cfg_shift (cfg_shift),
        .w_we (w_we), .w_out_ch (w_out_ch), .w_in_ch (w_in_ch), .w_data (w_data),
        .s_data (s_data), .s_valid (s_valid), .s_ready (s_ready),
        .m_data (conv_data), .m_sat (conv_sat),
        .m_valid (conv_valid), .m_ready (conv_ready), .m_last (conv_last)
    );

    act_unit #(
        .DATA_W (DATA_W), .LANES (N_OUT), .MODE (ACT_MODE)
    ) u_act (
        .linear (cfg_linear), .d_in (conv_data), .d_out (act_data)
    );

    always_ff @(posedge clk) begin
        if (!rst_n || sat_clear)                     sat_sticky <= 1'b0;
        else if (conv_valid && conv_ready && |conv_sat) sat_sticky <= 1'b1;
    end

    axis_serializer #(
        .DATA_W (DATA_W), .LANES (N_OUT)
    ) u_ser (
        .clk (clk), .rst_n (rst_n),
        .s_data (act_data), .s_last (conv_last),
        .s_valid (conv_valid), .s_ready (conv_ready),
        .m_data (m_data), .m_last (m_last),
        .m_valid (m_valid), .m_ready (m_ready)
    );

endmodule
