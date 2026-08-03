module input_adapter #(
    parameter PIX_WIDTH = 8,
    parameter ACT_WIDTH = 8
)(
    input  wire                     clk,
    input  wire                     rst,

    input  wire                     pixel_valid,
    input  wire [PIX_WIDTH-1:0]     pixel_in,      // image pixel (0-255)
    input  wire [PIX_WIDTH-1:0]     pixel_prev,    // previous frame pixel

    input  wire                     fsm_timestep_en,

    output reg  signed [ACT_WIDTH-1:0] input_act,
    output reg                      enable
);
    wire signed [ACT_WIDTH-1:0] diff_pixel;

    // Frame difference (hybrid input)
    assign diff_pixel = $signed({1'b0, pixel_in}) -
                        $signed({1'b0, pixel_prev});
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            input_act <= 0;
            enable    <= 0;
        end
        else begin
            enable <= fsm_timestep_en && pixel_valid;
            if (fsm_timestep_en && pixel_valid)
                input_act <= diff_pixel;
        end
    end
endmodule
