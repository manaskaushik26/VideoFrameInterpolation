`timescale 1ns / 1ps

module lif_neuron #(
    parameter DATA_W   = 16,                 // Q1.15
    parameter signed [15:0] BETA   = 16'd29491,  // 0.9
    parameter signed [15:0] THRESH = 16'd16384,  // 0.5
    parameter signed [15:0] V_RESET = 16'd0
)(
    input  wire                    clk,
    input  wire                    rst,
    input  wire                    valid_in,
    input  wire signed [DATA_W-1:0] input_current,
    output reg                     spike,
    output reg  signed [DATA_W-1:0] mem
);

    // Intermediate for beta * mem (Q1.15 × Q1.15 = Q2.30)
    wire signed [31:0] mem_mult;
    assign mem_mult = mem * BETA;

    // Back to Q1.15
    wire signed [DATA_W-1:0] mem_decay;
    assign mem_decay = mem_mult >>> 15;

    always @(posedge clk) begin
        if (rst) begin
            mem   <= V_RESET;
            spike <= 1'b0;
        end else if (valid_in) begin
            // Integrate
            mem <= mem_decay + input_current;
           
            // Fire
            if (mem_decay + input_current >= THRESH) begin
                spike <= 1'b1;
                mem   <= V_RESET;
            end else begin
                spike <= 1'b0;
            end
        end else begin
            spike <= 1'b0;
        end
    end

endmodule
