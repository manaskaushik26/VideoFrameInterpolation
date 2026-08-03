`timescale 1ns / 1ps

module top #(
    parameter WIDTH     = 224,
    parameter HEIGHT    = 224,
    parameter CHANNELS  = 6,
    parameter PIXEL_W   = 8,

    parameter SRC_W     = 32,
    parameter DST_W     = 8,
    parameter WEIGHT_W  = 16
)(
    input  wire                     clk,
    input  wire                     rst_n,

    // AXI-Stream input (DMA → PL)
    input  wire [PIXEL_W-1:0]       s_axis_tdata,
    input  wire                     s_axis_tvalid,
    input  wire                     s_axis_tlast,
    output wire                     s_axis_tready,

    // Final output
    output wire                     out_valid,
    output wire [DST_W-1:0]         out_neuron_id,
    output wire signed [15:0]       out_value
);

    /*--------------------------------------------------
      Reset
    --------------------------------------------------*/
    wire rst = ~rst_n;

    /*--------------------------------------------------
      Global FSM
    --------------------------------------------------*/
    wire timestep_en;
    wire array_reset;
    wire frame_done;

    fsm_global u_fsm_global (
        .clk         (clk),
        .rst         (rst),
        .start       (1'b1),
        .timestep_en (timestep_en),
        .array_reset (array_reset),
        .frame_done  (frame_done)
    );

    /*--------------------------------------------------
      AXI stream → pixel
    --------------------------------------------------*/
    assign s_axis_tready = 1'b1;

    wire [PIXEL_W-1:0] pixel_in    = s_axis_tdata;
    wire               pixel_valid = s_axis_tvalid;

    /*--------------------------------------------------
      Frame Assembler
    --------------------------------------------------*/
    localparam FB_ADDR_W = $clog2(WIDTH * HEIGHT * CHANNELS);

    wire [FB_ADDR_W-1:0] fb_addr;
    wire [PIXEL_W-1:0]   fb_din;
    wire                 fb_we;

    frame_assembler #(
        .WIDTH    (WIDTH),
        .HEIGHT   (HEIGHT),
        .CHANNELS (CHANNELS),
        .PIXEL_W  (PIXEL_W)
    ) u_frame_assembler (
        .clk         (clk),
        .rst         (rst),
        .pixel_in    (pixel_in),
        .pixel_valid (pixel_valid),
        .fb_addr     (fb_addr),
        .fb_din      (fb_din),
        .fb_we       (fb_we),
        .frame_done  (frame_done)
    );

    /*--------------------------------------------------
      Frame Buffer
    --------------------------------------------------*/
    wire [PIXEL_W-1:0] pixel_prev;

    frame_buffer u_frame_buffer (
        .clk      (clk),
        .fb_addr  (fb_addr),
        .fb_din   (fb_din),
        .fb_we    (fb_we),
        .rd_addr  (fb_addr),
        .rd_data  (pixel_prev)
    );

    /*--------------------------------------------------
      Input Adapter
    --------------------------------------------------*/
    wire signed [PIXEL_W-1:0] input_act;
    wire                     input_enable;

    input_adapter u_input_adapter (
        .clk             (clk),
        .rst             (rst),
        .pixel_valid     (pixel_valid),
        .pixel_in        (pixel_in),
        .pixel_prev      (pixel_prev),
        .fsm_timestep_en (timestep_en),
        .input_act       (input_act),
        .enable          (input_enable)
    );

    /*--------------------------------------------------
      Spike Router
    --------------------------------------------------*/
    wire                      route_valid;
    wire [DST_W-1:0]          route_dst_id;
    wire signed [WEIGHT_W-1:0] route_weight;

    spike_router u_spike_router (
        .clk           (clk),
        .rst           (rst),
        .spike_valid   (input_enable),
        .spike_src_id  (fb_addr),   // full linear index
        .route_valid   (route_valid),
        .route_dst_id  (route_dst_id),
        .route_weight  (route_weight),
        .bram_addr     (),
        .bram_dout     ()
    );

    /*--------------------------------------------------
      Synapse FIFO
    --------------------------------------------------*/
    localparam FIFO_W = DST_W + WEIGHT_W;

    wire [FIFO_W-1:0] fifo_dout;
    wire              fifo_empty;
    wire              fifo_rd_en;

    synapse_fifo #(
        .DATA_W (FIFO_W),
        .DEPTH  (16)
    ) u_synapse_fifo (
        .clk   (clk),
        .rst   (rst),
        .wr_en (route_valid),
        .din   ({route_dst_id, route_weight}),
        .full  (),
        .rd_en (fifo_rd_en),
        .dout  (fifo_dout),
        .empty (fifo_empty)
    );

    assign fifo_rd_en = ~fifo_empty;

    wire [DST_W-1:0]          fifo_dst_id  = fifo_dout[FIFO_W-1:WEIGHT_W];
    wire signed [WEIGHT_W-1:0] fifo_weight = fifo_dout[WEIGHT_W-1:0];

    /*--------------------------------------------------
      Synapse Accumulator
    --------------------------------------------------*/
    wire signed [15:0] syn_current;

    synapse_accumulator u_synapse_accumulator (
        .clk         (clk),
        .rst         (rst),
        .enable      (~fifo_empty),
        .clear_acc   (array_reset),
        .input_act   (input_act),
        .weight      (fifo_weight),
        .syn_current (syn_current)
    );

    /*--------------------------------------------------
      LIF Neuron
    --------------------------------------------------*/
    wire spike;
    wire valid_out;

    lif_neuron u_lif_neuron (
        .clk           (clk),
        .rst           (rst),
        .timestep_en   (timestep_en),
        .valid_in      (~fifo_empty),
        .input_current (syn_current),
        .spike         (spike),
        .mem           (),
        .valid_out     (valid_out)
    );

    /*--------------------------------------------------
      Readout
    --------------------------------------------------*/
    readout u_readout (
        .clk          (clk),
        .rst          (rst),
        .spike_in     (spike),
        .timestep_en  (timestep_en),
        .clear_window (array_reset),
        .window_done  (frame_done),
        .pixel_value  (out_value)
    );

    assign out_valid     = valid_out;
    assign out_neuron_id = fifo_dst_id;

endmodule
