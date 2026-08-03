module fsm_global_tb;

    reg clk = 0;
    always #5 clk = ~clk;

    reg rst;
    reg start;

    wire timestep_en;
    wire array_reset;
    wire frame_done;

    fsm_global #(.PIXELS_PER_FRAME(4)) dut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .timestep_en(timestep_en),
        .array_reset(array_reset),
        .frame_done(frame_done)
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
