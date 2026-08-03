module axi_pixel_adapter #(
    parameter PIX_WIDTH    = 8,
    parameter NUM_CHANNELS = 6,
    parameter NUM_BITS     = 8
)(
    input  wire                 clk,
    input  wire                 rst_n,

    // AXI-Stream slave
    input  wire [31:0]          s_axis_tdata,
    input  wire                 s_axis_tvalid,
    output wire                 s_axis_tready,

    // Outputs to SNN pipeline
    output reg  [PIX_WIDTH-1:0] pixel_out,
    output reg                  pixel_valid,
    output reg  [$clog2(NUM_CHANNELS)-1:0] channel_id,
    output reg  [$clog2(NUM_BITS)-1:0]     bit_id
);

    assign s_axis_tready = 1'b1;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pixel_out   <= '0;
            pixel_valid <= 1'b0;
            channel_id  <= '0;
            bit_id      <= '0;
        end else begin
            if (s_axis_tvalid) begin
                pixel_out   <= s_axis_tdata[PIX_WIDTH-1:0];
                pixel_valid <= 1'b1;

                // bit increments fastest
                if (bit_id == NUM_BITS-1) begin
                    bit_id <= 0;
                    if (channel_id == NUM_CHANNELS-1)
                        channel_id <= 0;
                    else
                        channel_id <= channel_id + 1'b1;
                end else begin
                    bit_id <= bit_id + 1'b1;
                end
            end else begin
                pixel_valid <= 1'b0;
            end
        end
    end

endmodule
