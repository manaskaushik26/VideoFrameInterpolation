`timescale 1ns / 1ps

module tb_readout;

    // Parameters
    localparam CNT_WIDTH = 8;

    // Signals
    reg                  clk;
    reg                  rst;
    reg                  spike_in;
    reg                  timestep_en;
    reg                  clear_window;
    reg                  window_done;

    wire [CNT_WIDTH-1:0] pixel_value;

    // DUT instantiation
    readout #(
        .CNT_WIDTH(CNT_WIDTH)
    ) dut (
        .clk(clk),
        .rst(rst),
        .spike_in(spike_in),
        .timestep_en(timestep_en),
        .clear_window(clear_window),
        .window_done(window_done),
        .pixel_value(pixel_value)
    );

    // Clock generation (10ns period)
    always #5 clk = ~clk;

    // Test sequence
    initial begin
        // Initialize
        clk = 0;
        rst = 1;
        spike_in = 0;
        timestep_en = 0;
        clear_window = 0;
        window_done = 0;

        // Hold reset
        #20;
        rst = 0;

        // Start new window
        @(posedge clk);
        clear_window = 1;
        @(posedge clk);
        clear_window = 0;

        // Simulate 10 timesteps
        // Spike pattern: 1 0 1 1 0 0 1 0 0 1  → total = 5 spikes
        repeat (10) begin
            @(posedge clk);
            timestep_en = 1;
            spike_in = $random % 2;  // random spike (0 or 1)
        end

        // Stop counting
        @(posedge clk);
        timestep_en = 0;
        spike_in = 0;
        window_done = 1;

        @(posedge clk);
        window_done = 0;

        // End simulation
        #20;
        $display("Final pixel_value = %d", pixel_value);
        $stop;
    end

endmodule
