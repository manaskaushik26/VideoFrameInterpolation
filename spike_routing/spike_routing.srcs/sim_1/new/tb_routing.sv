`timescale 1ns/1ps

module tb_spike_router;

  // -------------------------------------------------
  // Parameters (match DUT)
  // -------------------------------------------------
  localparam SRC_W    = 8;
  localparam DST_W    = 8;
  localparam WEIGHT_W = 16;
  localparam FANOUT   = 9;
  localparam ADDR_W   = 16;

  // -------------------------------------------------
  // DUT signals
  // -------------------------------------------------
  logic clk;
  logic rst;

  logic spike_valid;
  logic [SRC_W-1:0] spike_src_id;

  logic route_valid;
  logic [DST_W-1:0] route_dst_id;
  logic signed [WEIGHT_W-1:0] route_weight;

  logic [ADDR_W-1:0] bram_addr;
  logic [DST_W+WEIGHT_W-1:0] bram_dout;

  // -------------------------------------------------
  // Instantiate DUT
  // -------------------------------------------------
  spike_router #(
    .SRC_W(SRC_W),
    .DST_W(DST_W),
    .WEIGHT_W(WEIGHT_W),
    .FANOUT(FANOUT),
    .ADDR_W(ADDR_W)
  ) dut (
    .clk(clk),
    .rst(rst),
    .spike_valid(spike_valid),
    .spike_src_id(spike_src_id),
    .route_valid(route_valid),
    .route_dst_id(route_dst_id),
    .route_weight(route_weight),
    .bram_addr(bram_addr),
    .bram_dout(bram_dout)
  );

  // -------------------------------------------------
  // Clock generation (100 MHz)
  // -------------------------------------------------
  always #5 clk = ~clk;

  // -------------------------------------------------
  // Simple BRAM model
  // -------------------------------------------------
  logic [DST_W+WEIGHT_W-1:0] bram_mem [0:1023];

  always_comb begin
    bram_dout = bram_mem[bram_addr];
  end

  // -------------------------------------------------
  // Test stimulus
  // -------------------------------------------------
  integer i;
  localparam TEST_SRC = 8'd3;

  initial begin
    // Init
    clk = 0;
    rst = 1;
    spike_valid = 0;
    spike_src_id = 0;

    // Init routing table
    // For src_id = 3, fanout entries:
    // dst = 10+i, weight = i*100
    for (i = 0; i < FANOUT; i++) begin
      bram_mem[TEST_SRC*FANOUT + i] =
        {8'(10+i), 16'(i*100)};
    end

    // Reset
    #20;
    rst = 0;

    // Send spike
    @(posedge clk);
    spike_src_id = TEST_SRC;
    spike_valid  = 1'b1;

    @(posedge clk);
    spike_valid  = 1'b0;

    // Wait for routing to complete
    repeat (20) @(posedge clk);

    $display("Spike routing TB completed");
    $finish;
  end

  // -------------------------------------------------
  // Monitor routed outputs
  // -------------------------------------------------
  always_ff @(posedge clk) begin
    if (route_valid) begin
      $display("[%0t] ROUTE: dst=%0d weight=%0d addr=%0d",
               $time, route_dst_id, route_weight, bram_addr);
    end
  end

endmodule
