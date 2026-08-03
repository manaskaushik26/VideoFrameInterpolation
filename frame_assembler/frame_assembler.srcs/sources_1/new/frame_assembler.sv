module frame_assembler #(
    parameter WIDTH      = 224,   // image width
    parameter HEIGHT     = 224,   // image height
    parameter PIXEL_W    = 8,
    parameter ADDR_W     = 16
)(
    input  wire                 clk,
    input  wire                 rst,

    // From readout
    input  wire [PIXEL_W-1:0]   pixel_in,
    input  wire                 pixel_valid,   // == window_done

    // Frame buffer write interface
    output logic [ADDR_W-1:0]   fb_addr,
    output logic [PIXEL_W-1:0]  fb_din,
    output logic                fb_we,

    // Frame status
    output logic                frame_done
);

    // -------------------------------------------------
    // X-Y counters (spatial indexing happens HERE)
    // -------------------------------------------------
    logic [$clog2(WIDTH)-1:0]  x;
    logic [$clog2(HEIGHT)-1:0] y;

    // -------------------------------------------------
    // Raster-scan indexing
    // -------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            x <= '0;
            y <= '0;
        end
        else if (pixel_valid) begin
            if (x == WIDTH-1) begin
                x <= '0;
                if (y == HEIGHT-1)
                    y <= '0;
                else
                    y <= y + 1'b1;
            end
            else begin
                x <= x + 1'b1;
            end
        end
    end

    // -------------------------------------------------
    // Frame buffer write logic
    // -------------------------------------------------
    always_comb begin
        fb_addr = (y * WIDTH) + x;
        fb_din  = pixel_in;
        fb_we   = pixel_valid;
    end

    // -------------------------------------------------
    // Frame completion flag
    // -------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst)
            frame_done <= 1'b0;
        else
            frame_done <= pixel_valid &&
                          (x == WIDTH-1) &&
                          (y == HEIGHT-1);
    end

endmodule
