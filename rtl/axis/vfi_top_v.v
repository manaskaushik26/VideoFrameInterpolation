`timescale 1ns / 1ps
//
// vfi_top_v -- plain Verilog-2001 wrapper around vfi_top.
//
// Exists only because Vivado refuses a SystemVerilog file as the top of a
// block-design module reference:
//
//   [filemgmt 56-195] Reference 'vfi_top' contains top file '...vfi_top.sv' of
//   type SystemVerilog. This type is not allowed as the top file in the
//   reference.
//
// So the block design instantiates this, and this instantiates the real thing.
// It adds no logic. Keep the port list identical to vfi_top's, keep the names
// (aclk / aresetn / s_axi_* / s_axis_* / m_axis_* / s_wt_*) so the BD infers the
// interfaces, and keep it Verilog-2001 -- no logic type, no always_ff, nothing
// that would make Vivado classify it as SystemVerilog again.
//
module vfi_top_v #(
    parameter DATA_W   = 16,
    parameter WEIGHT_W = 16,
    parameter IMG_W    = 112,
    parameter IMG_H    = 112,
    parameter MAX_CH   = 128,
    parameter N_OUT    = 24,
    parameter ACC_W    = 48,
    parameter ACT_MODE = 1
)(
    input  wire                  aclk,
    input  wire                  aresetn,

    input  wire [4:0]            s_axi_awaddr,
    input  wire                  s_axi_awvalid,
    output wire                  s_axi_awready,
    input  wire [31:0]           s_axi_wdata,
    input  wire [3:0]            s_axi_wstrb,
    input  wire                  s_axi_wvalid,
    output wire                  s_axi_wready,
    output wire [1:0]            s_axi_bresp,
    output wire                  s_axi_bvalid,
    input  wire                  s_axi_bready,
    input  wire [4:0]            s_axi_araddr,
    input  wire                  s_axi_arvalid,
    output wire                  s_axi_arready,
    output wire [31:0]           s_axi_rdata,
    output wire [1:0]            s_axi_rresp,
    output wire                  s_axi_rvalid,
    input  wire                  s_axi_rready,

    input  wire [WEIGHT_W-1:0]   s_wt_tdata,
    input  wire                  s_wt_tvalid,
    output wire                  s_wt_tready,

    input  wire [DATA_W-1:0]     s_axis_tdata,
    input  wire                  s_axis_tlast,
    input  wire                  s_axis_tvalid,
    output wire                  s_axis_tready,

    output wire [DATA_W-1:0]     m_axis_tdata,
    output wire                  m_axis_tlast,
    output wire                  m_axis_tvalid,
    input  wire                  m_axis_tready
);

    vfi_top #(
        .DATA_W (DATA_W), .WEIGHT_W (WEIGHT_W),
        .IMG_W (IMG_W), .IMG_H (IMG_H),
        .MAX_CH (MAX_CH), .N_OUT (N_OUT),
        .ACC_W (ACC_W), .ACT_MODE (ACT_MODE)
    ) u_vfi (
        .aclk (aclk), .aresetn (aresetn),
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
        .s_wt_tdata (s_wt_tdata), .s_wt_tvalid (s_wt_tvalid),
        .s_wt_tready (s_wt_tready),
        .s_axis_tdata (s_axis_tdata), .s_axis_tlast (s_axis_tlast),
        .s_axis_tvalid (s_axis_tvalid), .s_axis_tready (s_axis_tready),
        .m_axis_tdata (m_axis_tdata), .m_axis_tlast (m_axis_tlast),
        .m_axis_tvalid (m_axis_tvalid), .m_axis_tready (m_axis_tready)
    );

endmodule
