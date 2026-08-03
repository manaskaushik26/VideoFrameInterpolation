module top_snn;

    reg clk = 0;
    always #5 clk = ~clk;

    reg rst;
    reg start;

    wire timestep_en;
    wire array_reset;

    wire [3:0] spike_in;
    wire [3:0] spike_out;

    // Fake spike encoder for now
    assign spike_in = 4'b1010;

    global_fsm #(.PIXELS_PER_FRAME(4)) fsm (
        .clk(clk),
        .rst(rst),
        .start(start),
        .timestep_en(timestep_en),
        .array_reset(array_reset),
        .frame_done()
    );

    neuron_array #(.N_NEURONS(4)) array (
        .clk(clk),
        .rst(array_reset),
        .timestep_en(timestep_en),
        .spike_in(spike_in),
        .spike_out(spike_out)
    );

    initial begin
        rst   = 1;
        start = 0;
        #20 rst = 0;
        #20 start = 1;
        #10 start = 0;
        #300 $finish;
    end

endmodule
