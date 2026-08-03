`timescale 1ns/1ps

module tb_synapse_fifo;

    // -----------------------------
    // Parameters
    // -----------------------------
    localparam DATA_W = 24;
    localparam DEPTH  = 16;

    // -----------------------------
    // Signals
    // -----------------------------
    logic              clk;
    logic              rst;

    logic              wr_en;
    logic [DATA_W-1:0] din;
    wire               full;

    logic              rd_en;
    wire [DATA_W-1:0]  dout;
    wire               empty;

    // -----------------------------
    // DUT instantiation
    // -----------------------------
    synapse_fifo #(
        .DATA_W(DATA_W),
        .DEPTH (DEPTH)
    ) dut (
        .clk   (clk),
        .rst   (rst),
        .wr_en (wr_en),
        .din   (din),
        .full  (full),
        .rd_en (rd_en),
        .dout  (dout),
        .empty (empty)
    );

    // -----------------------------
    // Clock generation (10ns period)
    // -----------------------------
    always #5 clk = ~clk;

    // -----------------------------
    // Test sequence
    // -----------------------------
    initial begin
        // Init
        clk   = 0;
        rst   = 1;
        wr_en = 0;
        rd_en = 0;
        din   = '0;

        // Hold reset
        #20;
        rst = 0;

        // ---------------------------------
        // 1️⃣ Write 5 values into FIFO
        // ---------------------------------
        repeat (5) begin
            @(posedge clk);
            wr_en = 1;
            din   = din + 24'h1;
        end

        @(posedge clk);
        wr_en = 0;

        // ---------------------------------
        // 2️⃣ Read 5 values from FIFO
        // ---------------------------------
        repeat (5) begin
            @(posedge clk);
            rd_en = 1;
        end

        @(posedge clk);
        rd_en = 0;

        // ---------------------------------
        // 3️⃣ Finish simulation
        // ---------------------------------
        #50;
        $finish;
    end

endmodule
