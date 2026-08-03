module input_adapter #(
    parameter PIX_WIDTH    = 8,
    parameter ACT_WIDTH    = 8,
    parameter WIDTH        = 224,
    parameter HEIGHT       = 224,
    parameter NUM_CHANNELS = 6,
    parameter NUM_BITS     = 8
)(
    input  wire                     clk,
    input  wire                     rst,

    input  wire                     pixel_valid,
    input  wire [PIX_WIDTH-1:0]     pixel_in,
    input  wire [PIX_WIDTH-1:0]     pixel_prev,

    input  wire [$clog2(NUM_CHANNELS)-1:0] channel_id,
    input  wire [$clog2(NUM_BITS)-1:0]     bit_id,

    input  wire                     fsm_timestep_en,

    output reg  signed [ACT_WIDTH-1:0] input_act,
    output reg                      enable,
    output reg  [31:0]               src_neuron_id
);

    wire signed [ACT_WIDTH-1:0] diff_pixel;

    assign diff_pixel =
        $signed({1'b0, pixel_in}) -
        $signed({1'b0, pixel_prev});

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            input_act     <= 0;
            enable        <= 0;
            src_neuron_id <= 0;
        end else begin
            enable <= fsm_timestep_en && pixel_valid;

            if (fsm_timestep_en && pixel_valid) begin
                input_act <= diff_pixel;

                // neuron indexing:
                // (((y * WIDTH + x) * NUM_CHANNELS + channel) * NUM_BITS + bit)
                src_neuron_id <=
                    (channel_id * NUM_BITS) + bit_id;
            end
        end
    end

endmodule
