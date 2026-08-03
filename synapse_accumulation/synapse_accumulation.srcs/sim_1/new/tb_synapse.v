`timescale 1ns / 1ps

module tb_synapse_accumulator;

    // Parameters
    parameter IN_WIDTH  = 8;
    parameter W_WIDTH   = 8;
    parameter ACC_WIDTH = 24;

    // DUT signals
    reg  clk;
    reg  rst;
    reg  enable;
    reg  clear_acc;
    reg  signed [IN_WIDTH-1:0]  input_act;
    reg  signed [W_WIDTH-1:0]   weight;
    wire signed [ACC_WIDTH-1:0] syn_current;

    // DUT instantiation
    synapse_accumulator #(
        .IN_WIDTH(IN_WIDTH),
        .W_WIDTH(W_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) dut (
        .clk(clk),
        .rst(rst),
        .enable(enable),
        .clear_acc(clear_acc),
        .input_act(input_act),
        .weight(weight),
        .syn_current(syn_current)
    );

    // Clock generation: 100 MHz
    always #5 clk = ~clk;

    initial begin
        // -------------------------
        // INITIALIZE
        // -------------------------
        clk       = 0;
        rst       = 1;
        enable    = 0;
        clear_acc = 0;
        input_act = 0;
        weight    = 0;

        // -------------------------
        // RESET
        // -------------------------
        #20;
        rst = 0;

        // -------------------------
        // TEST 1: Clear + first MAC
        // 8 × 3 = 24
        // -------------------------
        clear_acc = 1;
        #10;
        clear_acc = 0;

        input_act = 8;
        weight    = 3;
        enable    = 1;
        #10;

        // -------------------------
        // TEST 2: Accumulate again
        // 4 × 2 = 8 → total = 32
        // -------------------------
        input_act = 4;
        weight    = 2;
        #10;

        // -------------------------
        // TEST 3: Signed weight
        // 5 × (-1) = -5 → total = 27
        // -------------------------
        input_act = 5;
        weight    = -1;
        #10;

        enable = 0;

        // -------------------------
        // TEST 4: Clear accumulator
        // -------------------------
        clear_acc = 1;
        #10;
        clear_acc = 0;

        // -------------------------
        // TEST 5: Negative input
        // -6 × 4 = -24
        // -------------------------
        input_act = -6;
        weight    = 4;
        enable    = 1;
        #10;

        enable = 0;

        // -------------------------
        // END SIMULATION
        // -------------------------
        #20;
        $stop;
    end

endmodule
