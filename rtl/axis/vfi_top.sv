`timescale 1ns / 1ps
//
// vfi_top -- the IP the block design instantiates.
//
//     s_axi    control and status, AXI4-Lite (PS)
//     s_wt     kernels for the current layer,  AXI4-Stream (DMA MM2S)
//     s_axis   input feature map,              AXI4-Stream (DMA MM2S)
//     m_axis   output feature map,             AXI4-Stream (DMA S2MM)
//
// One convolution layer, configured at runtime. The PS runs the network by
// repeating, per layer:
//
//     write CFG (channels, shift)  ->  DMA kernels  ->  DMA feature map in
//                                  ->  DMA feature map out  ->  read STATUS
//
// The output of one layer is the input of the next, in the same
// (row, col, channel) channel-minor layout, so it goes straight back to DDR and
// comes back as the next layer's input with no reformatting.
//
// This replaces top_axis.sv / vfi_top_0, which had no AXI-Lite interface at
// all, violated AXI-Stream six ways, and hung the DMA by never delivering TLAST
// to S2MM.
//
// Reset: aresetn is the block design's active-low reset. soft_reset from the
// control register is ORed into the datapath reset only -- it deliberately does
// not reset axil_regs itself, or the write that asserted it would clear the
// register block mid-transaction and drop the AXI-Lite response, hanging the PS.
//
module vfi_top #(
    parameter int DATA_W    = 16,
    parameter int WEIGHT_W  = 16,
    parameter int IMG_W     = 112,
    parameter int IMG_H     = 112,
    parameter int MAX_CH    = 128,
    parameter int N_OUT     = 24,
    parameter int ACC_W     = 48,
    parameter int ACT_MODE  = 1
)(
    input  logic                  aclk,
    input  logic                  aresetn,

    // AXI4-Lite control
    input  logic [4:0]            s_axi_awaddr,
    input  logic                  s_axi_awvalid,
    output logic                  s_axi_awready,
    input  logic [31:0]           s_axi_wdata,
    input  logic [3:0]            s_axi_wstrb,
    input  logic                  s_axi_wvalid,
    output logic                  s_axi_wready,
    output logic [1:0]            s_axi_bresp,
    output logic                  s_axi_bvalid,
    input  logic                  s_axi_bready,
    input  logic [4:0]            s_axi_araddr,
    input  logic                  s_axi_arvalid,
    output logic                  s_axi_arready,
    output logic [31:0]           s_axi_rdata,
    output logic [1:0]            s_axi_rresp,
    output logic                  s_axi_rvalid,
    input  logic                  s_axi_rready,

    // kernels
    input  logic [WEIGHT_W-1:0]   s_wt_tdata,
    input  logic                  s_wt_tvalid,
    output logic                  s_wt_tready,

    // feature map in
    input  logic [DATA_W-1:0]     s_axis_tdata,
    input  logic                  s_axis_tlast,
    input  logic                  s_axis_tvalid,
    output logic                  s_axis_tready,

    // feature map out
    output logic [DATA_W-1:0]     m_axis_tdata,
    output logic                  m_axis_tlast,
    output logic                  m_axis_tvalid,
    input  logic                  m_axis_tready
);

    localparam int CFGW = $clog2(MAX_CH + 1);

    // Ports are named for Vivado's block-design interface inference: aclk /
    // aresetn associate the clock and reset automatically, s_axi_* is recognised
    // as AXI4-Lite, and s_axis_/m_axis_/s_wt_ as AXI4-Stream. Renaming these
    // breaks that inference and the BD cell comes up with loose pins instead of
    // interfaces, which then will not connect to the DMA or the interconnect.
    logic rst_n_i;
    assign rst_n_i = aresetn;

    logic [7:0] cfg_channels_r;
    logic [5:0] cfg_shift_r;
    logic       sat_clear, soft_reset;
    logic       weights_loaded, sat_sticky;

    // One pulse per completed output frame, for the FRAMES counter. Taken from
    // the output TLAST handshake, which is the only unambiguous end-of-frame.
    logic frame_tick;
    assign frame_tick = m_axis_tvalid && m_axis_tready && m_axis_tlast;

    axil_regs #(
        .ADDR_W (5), .CH_W (8), .SH_W (6)
    ) u_regs (
        .clk (aclk), .rst_n (rst_n_i),
        .s_axi_awaddr (s_axi_awaddr), .s_axi_awvalid (s_axi_awvalid),
        .s_axi_awready (s_axi_awready),
        .s_axi_wdata (s_axi_wdata), .s_axi_wstrb (s_axi_wstrb),
        .s_axi_wvalid (s_axi_wvalid), .s_axi_wready (s_axi_wready),
        .s_axi_bresp (s_axi_bresp), .s_axi_bvalid (s_axi_bvalid),
        .s_axi_bready (s_axi_bready),
        .s_axi_araddr (s_axi_araddr), .s_axi_arvalid (s_axi_arvalid),
        .s_axi_arready (s_axi_arready),
        .s_axi_rdata (s_axi_rdata), .s_axi_rresp (s_axi_rresp),
        .s_axi_rvalid (s_axi_rvalid), .s_axi_rready (s_axi_rready),
        .cfg_channels (cfg_channels_r), .cfg_shift (cfg_shift_r),
        .sat_clear (sat_clear), .soft_reset (soft_reset),
        .weights_loaded (weights_loaded), .sat_sticky (sat_sticky),
        .frame_tick (frame_tick)
    );

    // Datapath reset only; see header.
    logic dp_rst_n;
    assign dp_rst_n = rst_n_i && !soft_reset;

    conv_stage_axis #(
        .DATA_W (DATA_W), .WEIGHT_W (WEIGHT_W), .IMG_W (IMG_W), .IMG_H (IMG_H),
        .MAX_CH (MAX_CH), .N_OUT (N_OUT), .ACC_W (ACC_W), .ACT_MODE (ACT_MODE)
    ) u_stage (
        .clk (aclk), .rst_n (dp_rst_n),
        .cfg_channels (CFGW'(cfg_channels_r)),
        .cfg_shift (cfg_shift_r),
        .s_wt_tdata (s_wt_tdata), .s_wt_tvalid (s_wt_tvalid), .s_wt_tready (s_wt_tready),
        .weights_loaded (weights_loaded),
        .sat_clear (sat_clear), .sat_sticky (sat_sticky),
        .s_axis_tdata (s_axis_tdata), .s_axis_tlast (s_axis_tlast),
        .s_axis_tvalid (s_axis_tvalid), .s_axis_tready (s_axis_tready),
        .m_axis_tdata (m_axis_tdata), .m_axis_tlast (m_axis_tlast),
        .m_axis_tvalid (m_axis_tvalid), .m_axis_tready (m_axis_tready)
    );

endmodule
