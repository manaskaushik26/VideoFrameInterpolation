module frame_assembler #(
    parameter WIDTH    = 224,
    parameter HEIGHT   = 224,
    parameter CHANNELS = 6,
    parameter PIXEL_W  = 8
)(
    input  wire                 clk,
    input  wire                 rst,

    input  wire [PIXEL_W-1:0]   pixel_in,
    input  wire                 pixel_valid,

    output logic [ADDR_W-1:0]   fb_addr,
    output logic [PIXEL_W-1:0]  fb_din,
    output logic                fb_we,

    output logic                frame_done
);

    localparam ADDR_W = $clog2(WIDTH*HEIGHT*CHANNELS);

    logic [$clog2(WIDTH)-1:0]   x;
    logic [$clog2(HEIGHT)-1:0]  y;
    logic [$clog2(CHANNELS)-1:0] ch;

    always_ff @(posedge clk) begin
        if (rst) begin
            x <= 0; y <= 0; ch <= 0;
        end else if (pixel_valid) begin
            if (x == WIDTH-1) begin
                x <= 0;
                if (y == HEIGHT-1) begin
                    y <= 0;
                    if (ch == CHANNELS-1)
                        ch <= 0;
                    else
                        ch <= ch + 1;
                end else begin
                    y <= y + 1;
                end
            end else begin
                x <= x + 1;
            end
        end
    end

    always_comb begin
        fb_addr = ch * (WIDTH*HEIGHT)
                + y  * WIDTH
                + x;
        fb_din  = pixel_in;
        fb_we   = pixel_valid;
    end

    always_ff @(posedge clk) begin
        if (rst)
            frame_done <= 1'b0;
        else
            frame_done <= pixel_valid &&
                          (x  == WIDTH-1) &&
                          (y  == HEIGHT-1) &&
                          (ch == CHANNELS-1);
    end

endmodule
