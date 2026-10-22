`timescale 1ns / 1ps
//
// conv_stage_axis -- one convolution layer with AXI-Stream interfaces.
//
//     S_AXIS -> skid -> conv_stage -> skid -> M_AXIS
//
// This is the unit the AXI-DMA talks to. It replaces top_axis.sv, which
// violated the AXI-Stream protocol six ways and hung the DMA by never
// delivering TLAST to S2MM.
//
// Stream format
// -------------
// TDATA is one Q1.15 value per beat, in (row, col, channel) order with channel
// minor -- identical on both interfaces, which is what lets stages chain and
// what the PS must match when it lays out a feature map in DDR.
//
//   input  packet = IMG_W * IMG_H * CH_IN  beats, TLAST on the final beat
//   output packet = IMG_W * IMG_H * N_OUT  beats, TLAST on the final beat
//
// Exactly one TLAST per frame. The DMA's S2MM channel needs it to complete a
// transfer; without it the transfer never finishes, which is the hang in the
// old design.
//
// s_axis_tlast is intentionally not used to delimit anything internally: the
// frame length is a parameter, and the line buffer's own counters define the
// frame. Trusting an incoming TLAST would let a short or long DMA descriptor
// desynchronise the pipeline silently. It is accepted and ignored by design,
// not by accident.
//
// Width note: 16-bit TDATA matches Q1.15 one value per beat, which is the
// simple and correct starting point. Widening to 32 or 64 bits with packing is
// worthwhile later for DMA bandwidth -- a 112x112x64 map is 1.6 MB per
// transfer -- but it is an optimisation, and it needs the packing order to
// preserve the channel-minor layout above.
//
module conv_stage_axis #(
    parameter int DATA_W    = 16,
    parameter int WEIGHT_W  = 16,
    parameter int IMG_W     = 112,
    parameter int IMG_H     = 112,
    parameter int MAX_CH    = 8,
    parameter int N_OUT     = 4,
    parameter int ACC_W     = 48,
    parameter int SHIFT_W   = 6,
    parameter int ACT_MODE  = 1
)(
    input  logic                        clk,
    input  logic                        rst_n,

    input  logic [$clog2(MAX_CH+1)-1:0] cfg_channels,
    input  logic [SHIFT_W-1:0]          cfg_shift,
    input  logic                        cfg_linear,     // 1: bypass ReLU (flow layer)

    // AXI-Stream slave carrying this layer's kernels, DMA'd as one blob.
    // N_OUT*CH_IN*9 beats, in the flattened (out, in, kh, kw) order of the
    // conv*_weight_q15.bin files. `weights_loaded` rises when a full set has
    // landed and is the signal that the layer is safe to run.
    input  logic [WEIGHT_W-1:0]         s_wt_tdata,
    input  logic                        s_wt_tvalid,
    output logic                        s_wt_tready,
    output logic                        weights_loaded,

    // status
    input  logic                        sat_clear,
    output logic                        sat_sticky,

    // AXI-Stream slave (from DMA MM2S)
    input  logic [DATA_W-1:0]           s_axis_tdata,
    input  logic                        s_axis_tlast,
    input  logic                        s_axis_tvalid,
    output logic                        s_axis_tready,

    // AXI-Stream master (to DMA S2MM)
    output logic [DATA_W-1:0]           m_axis_tdata,
    output logic                        m_axis_tlast,
    output logic                        m_axis_tvalid,
    input  logic                        m_axis_tready
);

    logic                  w_we;
    logic [$clog2(N_OUT > 1 ? N_OUT : 2)-1:0] w_out_ch;
    logic [$clog2(MAX_CH > 1 ? MAX_CH : 2)-1:0] w_in_ch;
    logic [9*WEIGHT_W-1:0] w_data;

    weight_loader #(
        .WEIGHT_W (WEIGHT_W), .MAX_CH (MAX_CH), .N_OUT (N_OUT)
    ) u_wl (
        .clk (clk), .rst_n (rst_n),
        .cfg_channels (cfg_channels),
        .s_axis_tdata (s_wt_tdata), .s_axis_tvalid (s_wt_tvalid),
        .s_axis_tready (s_wt_tready),
        .w_we (w_we), .w_out_ch (w_out_ch), .w_in_ch (w_in_ch), .w_data (w_data),
        .loaded (weights_loaded)
    );

    logic [DATA_W-1:0] in_data;
    logic              in_last, in_valid, in_ready;

    axis_skid #(.DATA_W (DATA_W)) u_skid_in (
        .clk (clk), .rst_n (rst_n),
        .s_data (s_axis_tdata), .s_last (s_axis_tlast),
        .s_valid (s_axis_tvalid), .s_ready (s_axis_tready),
        .m_data (in_data), .m_last (in_last),
        .m_valid (in_valid), .m_ready (in_ready)
    );

    logic [DATA_W-1:0] out_data;
    logic              out_last, out_valid, out_ready;

    conv_stage #(
        .DATA_W (DATA_W), .WEIGHT_W (WEIGHT_W),
        .IMG_W (IMG_W), .IMG_H (IMG_H),
        .MAX_CH (MAX_CH), .N_OUT (N_OUT),
        .ACC_W (ACC_W), .SHIFT_W (SHIFT_W), .ACT_MODE (ACT_MODE)
    ) u_stage (
        .clk (clk), .rst_n (rst_n),
        .cfg_channels (cfg_channels), .cfg_shift (cfg_shift), .cfg_linear (cfg_linear),
        .w_we (w_we), .w_out_ch (w_out_ch), .w_in_ch (w_in_ch), .w_data (w_data),
        .s_data (in_data), .s_valid (in_valid), .s_ready (in_ready),
        .m_data (out_data), .m_last (out_last),
        .m_valid (out_valid), .m_ready (out_ready),
        .sat_clear (sat_clear), .sat_sticky (sat_sticky)
    );

    axis_skid #(.DATA_W (DATA_W)) u_skid_out (
        .clk (clk), .rst_n (rst_n),
        .s_data (out_data), .s_last (out_last),
        .s_valid (out_valid), .s_ready (out_ready),
        .m_data (m_axis_tdata), .m_last (m_axis_tlast),
        .m_valid (m_axis_tvalid), .m_ready (m_axis_tready)
    );

    // s_axis_tlast is deliberately unused; see the header. Reference it so the
    // intent is explicit and lint does not merely report it as dropped, which is
    // indistinguishable from the old design's accidental drop.
    logic unused_tlast;
    assign unused_tlast = in_last;

endmodule
