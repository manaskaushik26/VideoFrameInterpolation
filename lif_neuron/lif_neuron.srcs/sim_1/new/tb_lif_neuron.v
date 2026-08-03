`timescale 1ns/1ps

module lif_neuron_tb;

    // --------------------------------------------------
    // Clock & Reset
    // --------------------------------------------------
    reg clk = 0;
    reg rstn = 0;

    always #5 clk = ~clk;   // 100 MHz clock

    // --------------------------------------------------
    // AXI Stream INPUT (to neuron)
    // --------------------------------------------------
    reg        s_axis_tvalid;
    wire       s_axis_tready;
    reg [0:0]  s_axis_tdata;

    // --------------------------------------------------
    // AXI Stream OUTPUT (from neuron)
    // --------------------------------------------------
    wire       m_axis_tvalid;
    reg        m_axis_tready;
    wire [0:0] m_axis_tdata;

    // --------------------------------------------------
    // DUT
    // --------------------------------------------------
    lif_neuron dut (
        .clk(clk),
        .rstn(rstn),

        .s_axis_tvalid(s_axis_tvalid),
        .s_axis_tready(s_axis_tready),
        .s_axis_tdata (s_axis_tdata),

        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready),
        .m_axis_tdata (m_axis_tdata)
    );

    // --------------------------------------------------
    // Reset sequence
    // --------------------------------------------------
    initial begin
        s_axis_tvalid = 0;
        s_axis_tdata  = 0;
        m_axis_tready = 1;

        #20;
        rstn = 1;
    end

    // --------------------------------------------------
    // Stimulus: send spikes
    // --------------------------------------------------
    initial begin
        // Wait for reset release
        @(posedge rstn);
        @(posedge clk);

        // Send multiple spikes to charge membrane
        repeat (6) begin
            send_spike();
            #20;
        end

        // Wait
        #200;

        $finish;
    end

    // --------------------------------------------------
    // AXI backpressure test
    // --------------------------------------------------
    initial begin
        // Let things run normally first
        #120;

        // Apply backpressure
        m_axis_tready = 0;
        #60;

        // Release backpressure
        m_axis_tready = 1;
    end

    // --------------------------------------------------
    // Task: Send one AXI-stream spike
    // --------------------------------------------------
    task send_spike;
    begin
        @(posedge clk);
        s_axis_tdata  <= 1'b1;
        s_axis_tvalid <= 1'b1;

        // Wait until handshake completes
        wait (s_axis_tvalid && s_axis_tready);
        @(posedge clk);

        s_axis_tvalid <= 1'b0;
        s_axis_tdata  <= 1'b0;
    end
    endtask

endmodule
