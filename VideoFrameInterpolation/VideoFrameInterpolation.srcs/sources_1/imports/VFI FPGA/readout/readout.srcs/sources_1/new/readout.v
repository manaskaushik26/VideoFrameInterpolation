module top #(
    parameter PIXEL_W   = 8,
    parameter WIDTH     = 224,
    parameter HEIGHT    = 224,
    parameter CHANNELS  = 6,

    parameter SRC_W     = 32,
    parameter DST_W     = 8,
    parameter WEIGHT_W  = 16,
    parameter ACC_W     = 24
)(
    input  wire                   clk,
    input  wire                   rst_n,

    // AXI-Stream pixel input (1 word = 1 pixel)
    input  wire [PIXEL_W-1:0]     s_axis_tdata,
    input  wire                   s_axis_tvalid,
    output wire                   s_axis_tready,

    // Final output
    output wire                   out_valid,
    output wire [7:0]             out_value
);

    // -------------------------------------------------
    // Reset
    // -------------------------------------------------
    wire rst = ~rst_n;

    // -------------------------------------------------
    // Always ready (no backpressure yet)
    // -------------------------------------------------
    assign s_axis_tready = 1'b1;

    // -------------------------------------------------
    // Global FSM
    // -------------------------------------------------
    wire timestep_en;
    wire frame_done;

    fsm_global u_fsm_global (
        .clk         (clk),
        .rst         (rst),
        .start       (1'b1),
        .timestep_en (timestep_en),
        .array_reset (),
        .frame_done  (frame_done)
    );

    // -------------------------------------------------
    // Pixel stream (direct for now)
    // -------------------------------------------------
    wire [PIXEL_W-1:0] pixel_in    = s_axis_tdata;
    wire               pixel_valid = s_axis_tvalid;

    // -------------------------------------------------
    // Frame Assembler
    // -------------------------------------------------
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


    // -------------------------------------------------
    // Frame Buffer
    // -------------------------------------------------
    wire [PIXEL_W-1:0] pixel_prev;

frame_buffer #(
    .WIDTH   (WIDTH),
    .HEIGHT  (HEIGHT),
    .PIXEL_W (PIXEL_W),
    .ADDR_W  (FB_ADDR_W)
) u_frame_buffer (
    .clk     (clk),
    .fb_addr (fb_addr),
    .fb_din  (fb_din),
    .fb_we   (fb_we),
    .rd_addr (fb_addr),      // reading same pixel for prev-frame
    .rd_data (pixel_prev)
);


    // -------------------------------------------------
    // Input Adapter
    // -------------------------------------------------
    wire signed [PIXEL_W-1:0] input_act;
    wire                      input_enable;

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

    // -------------------------------------------------
    // Spike Router
    // -------------------------------------------------
    wire                      route_valid;
    wire [DST_W-1:0]          route_dst_id;
    wire signed [WEIGHT_W-1:0] route_weight;

    spike_router u_spike_router (
        .clk          (clk),
        .rst          (rst),
        .spike_valid  (input_enable),
        .spike_src_id (input_act[SRC_W-1:0]),
        .route_valid  (route_valid),
        .route_dst_id (route_dst_id),
        .route_weight (route_weight),
        .bram_addr    (),
        .bram_dout    ()
    );

    // -------------------------------------------------
    // Synapse Accumulator (DIRECT, NO FIFO)
    // -------------------------------------------------
    wire signed [ACC_W-1:0] syn_current;

    synapse_accumulator u_synapse_accumulator (
        .clk        (clk),
        .rst        (rst),
        .enable     (route_valid),
        .clear_acc  (frame_done),
        .input_act  (input_act),
        .weight     (route_weight),
        .syn_current(syn_current)
    );

    // -------------------------------------------------
    // LIF Neuron
    // -------------------------------------------------
    wire spike;
    wire valid_out;

    lif_neuron u_lif_neuron (
        .clk       (clk),
        .rst       (rst),
        .in_valid  (route_valid),
        .in_current(syn_current),
        .spike     (spike),
        .mem       (),
        .valid_out (valid_out)
    );

    // -------------------------------------------------
    // Readout (single-channel, correct)
    // -------------------------------------------------
    readout u_readout (
        .clk          (clk),
        .rst          (rst),
        .spike_in     (spike),
        .timestep_en  (timestep_en),
        .clear_window (frame_done),
        .window_done  (frame_done),
        .pixel_value  (out_value)
    );

    assign out_valid = valid_out;

endmodule
