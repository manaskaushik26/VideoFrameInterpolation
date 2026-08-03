module frame_buffer #(
    parameter WIDTH    = 224,
    parameter HEIGHT   = 224,
    parameter CHANNELS = 6,
    parameter PIXEL_W  = 8
)(
    input  wire                 clk,

    input  wire [ADDR_W-1:0]    fb_addr,
    input  wire [PIXEL_W-1:0]   fb_din,
    input  wire                 fb_we,

    input  wire [ADDR_W-1:0]    rd_addr,
    output logic [PIXEL_W-1:0]  rd_data
);

    localparam FRAME_SIZE = WIDTH * HEIGHT * CHANNELS;
    localparam ADDR_W     = $clog2(FRAME_SIZE);

    logic [PIXEL_W-1:0] mem [0:FRAME_SIZE-1];

    always_ff @(posedge clk) begin
        if (fb_we)
            mem[fb_addr] <= fb_din;

        rd_data <= mem[rd_addr];
    end

endmodule
