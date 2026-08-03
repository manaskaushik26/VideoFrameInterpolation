`timescale 1ns / 1ps

module lif_neuron #(
    parameter DATA_W = 16,                  // Q1.15
    parameter signed [DATA_W-1:0] BETA   = 16'd29491, // 0.9
    parameter signed [DATA_W-1:0] THRESH = 16'd16384, // 0.5
    parameter signed [DATA_W-1:0] V_RESET = 16'd0
)(
    input  wire                    clk,
    input  wire                    rst,

    // Control
    input  wire                    timestep_en,   // << FROM TIMESTEP FSM

    // Input current
    input  wire                    valid_in,
    input  wire signed [DATA_W-1:0] input_current,

    // Outputs
    output reg                     spike,
    output reg  signed [DATA_W-1:0] mem,
    output reg                     valid_out
);

    // Q1.15 × Q1.15 → Q2.30
    wire signed [31:0] mem_mult = mem * BETA;

    // Back to Q1.15
    wire signed [DATA_W-1:0] mem_decay = mem_mult >>> 15;

    wire signed [DATA_W-1:0] mem_next =
        mem_decay + input_current;

    always @(posedge clk) begin
        if (rst) begin
            mem       <= V_RESET;
            spike     <= 1'b0;
            valid_out <= 1'b0;
        end else if (timestep_en && valid_in) begin
            valid_out <= 1'b1;

            if (mem_next >= THRESH) begin
                spike <= 1'b1;
                mem   <= V_RESET;
            end else begin
                spike <= 1'b0;
                mem   <= mem_next;
            end
        end else begin
            spike     <= 1'b0;
            valid_out <= 1'b0;
        end
    end

endmodule

