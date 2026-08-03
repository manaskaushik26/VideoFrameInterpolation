module synapse_accumulator #(
    parameter IN_WIDTH  = 8,
    parameter W_WIDTH   = 8,
    parameter ACC_WIDTH = 24
)(
    input  wire                     clk,
    input  wire                     rst,
    input  wire                     enable,
    input  wire                     clear_acc,
    input  wire signed [IN_WIDTH-1:0]  input_act,
    input  wire signed [W_WIDTH-1:0]   weight,
    output reg  signed [ACC_WIDTH-1:0] syn_current
);

    // Saturation limits
    localparam signed [ACC_WIDTH-1:0] MAX_ACC =
        {1'b0, {(ACC_WIDTH-1){1'b1}}};

    localparam signed [ACC_WIDTH-1:0] MIN_ACC =
        {1'b1, {(ACC_WIDTH-1){1'b0}}};

    wire signed [IN_WIDTH+W_WIDTH-1:0] mult_out;
    wire signed [ACC_WIDTH-1:0] mult_ext;

    assign mult_out = input_act * weight;

    // Explicit sign extension
    assign mult_ext = {{(ACC_WIDTH-(IN_WIDTH+W_WIDTH)){mult_out[IN_WIDTH+W_WIDTH-1]}}, mult_out};

    always @(posedge clk or posedge rst) begin
        if (rst)
            syn_current <= 0;
        else if (clear_acc)
            syn_current <= 0;
        else if (enable) begin
            if (syn_current + mult_ext > MAX_ACC)
                syn_current <= MAX_ACC;
            else if (syn_current + mult_ext < MIN_ACC)
                syn_current <= MIN_ACC;
            else
                syn_current <= syn_current + mult_ext;
        end
    end

endmodule
