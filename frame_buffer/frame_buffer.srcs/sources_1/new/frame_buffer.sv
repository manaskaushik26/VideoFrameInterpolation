module frame_buffer #(
    parameter WIDTH   = 224,
    parameter HEIGHT  = 224,
    parameter PIXEL_W = 8,
    parameter ADDR_W  = 16              // enough for WIDTH*HEIGHT
)(
    input  wire                 clk,

    // Write interface (from frame_assembler)
    input  wire [ADDR_W-1:0]    fb_addr,
    input  wire [PIXEL_W-1:0]   fb_din,
    input  wire                 fb_we,

    // Read interface (to PS / AXI / debug)
    input  wire [ADDR_W-1:0]    rd_addr,
    output logic [PIXEL_W-1:0]  rd_data
);

    // -------------------------------------------------
    // Frame memory
    // -------------------------------------------------
    localparam FRAME_SIZE = WIDTH * HEIGHT;

    logic [PIXEL_W-1:0] mem [0:FRAME_SIZE-1];

    // -------------------------------------------------
    // Write pixel (PL side)
    // -------------------------------------------------
    always_ff @(posedge clk) begin
        if (fb_we) begin
            mem[fb_addr] <= fb_din;
        end
    end

    // -------------------------------------------------
    // Read pixel (PS / AXI side)
    // -------------------------------------------------
    always_ff @(posedge clk) begin
        rd_data <= mem[rd_addr];
    end

endmodule
