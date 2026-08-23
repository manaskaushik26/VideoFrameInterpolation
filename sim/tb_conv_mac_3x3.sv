`timescale 1ns / 1ps
//
// Self-checking testbench for conv_mac_3x3.
//
// The reference accumulates in 64-bit signed arithmetic, which cannot overflow
// for any input this module accepts, then applies the same round-and-saturate.
// Computing it at a width where wrapping is impossible is the point: it is the
// only way to catch a saturation path that is unreachable in the DUT, which is
// precisely the bug in the superseded synapse.v.
//
// Coverage:
//   - directed edge cases: zeros, +/- full scale, asymmetric saturation
//   - random windows/weights over random channel-run lengths
//   - a 128-channel run, the deepest layer in the network (conv3/conv4)
//   - random stalls on both s_valid and m_ready throughout
//
module tb_conv_mac_3x3;

    // Overridable so the sweep can exercise several requantisation shifts:
    //   run_xsim.ps1 ... -Defines @{TB_OUT_SHIFT=20}
`ifndef TB_OUT_SHIFT
  `define TB_OUT_SHIFT 15
`endif

    localparam int DATA_W    = 16;
    localparam int WEIGHT_W  = 16;
    localparam int ACC_W     = 48;
    localparam int OUT_SHIFT = `TB_OUT_SHIFT;

    localparam longint signed OUT_MAX =  (1 <<< (DATA_W-1)) - 1;
    localparam longint signed OUT_MIN = -(1 <<< (DATA_W-1));

    logic clk = 0;
    logic rst_n;
    always #5 clk = ~clk;

    logic [9*DATA_W-1:0]      s_window;
    logic [9*WEIGHT_W-1:0]    s_weights;
    logic                     s_first, s_last, s_tag, s_valid;
    logic                     s_ready;
    logic signed [DATA_W-1:0] m_data;
    logic                     m_sat, m_tag, m_valid, m_ready;

    conv_mac_3x3 #(
        .DATA_W (DATA_W), .WEIGHT_W (WEIGHT_W), .ACC_W (ACC_W), .OUT_SHIFT (OUT_SHIFT)
    ) dut (
        .clk (clk), .rst_n (rst_n),
        .s_window (s_window), .s_weights (s_weights),
        .s_first (s_first), .s_last (s_last), .s_tag (s_tag),
        .s_valid (s_valid), .s_ready (s_ready),
        .m_data (m_data), .m_sat (m_sat), .m_tag (m_tag),
        .m_valid (m_valid), .m_ready (m_ready)
    );

    int errors, checked;
    int stall_in, stall_out;

    // Coverage. A saturation path that never fires would let every test above
    // pass while the logic is unreachable -- which is the exact defect being
    // designed out, so it is counted rather than assumed.
    int n_sat_hi, n_sat_lo, n_exact, n_nonzero;

    // Expected results, in completion order.
    longint signed exp_data_q [$];
    bit               exp_sat_q  [$];
    string            exp_name_q [$];

    // Reference: round-half-up at FRAC, then clamp. Done in 64 bits so the
    // pre-saturation value is always exact.
    function automatic void ref_finish(longint signed acc, output longint signed d, output bit sat);
        longint signed r;
        begin
            if (OUT_SHIFT == 0) r = acc;
            else                r = (acc + (64'sd1 <<< (OUT_SHIFT-1))) >>> OUT_SHIFT;
            if      (r > OUT_MAX) begin d = OUT_MAX; sat = 1; end
            else if (r < OUT_MIN) begin d = OUT_MIN; sat = 1; end
            else                  begin d = r;       sat = 0; end
        end
    endfunction

    // Drive one pixel: nchan beats, first on 0, last on nchan-1. Windows and
    // weights come from the caller-supplied mode.
    //   mode 0 = random, 1 = all zero, 2 = all +full scale, 3 = all -full scale,
    //   mode 4 = max positive product (act and weight both most-negative)
    task automatic drive_pixel(string name, int nchan, int mode);
        logic signed [DATA_W-1:0]   a [9];
        logic signed [WEIGHT_W-1:0] w [9];
        longint signed acc;
        longint signed d;
        bit sat;
        begin
            acc = 0;
            for (int ch = 0; ch < nchan; ch++) begin
                for (int i = 0; i < 9; i++) begin
                    case (mode)
                        1: begin a[i] = 0;              w[i] = 0;              end
                        2: begin a[i] = OUT_MAX;        w[i] = OUT_MAX;        end
                        3: begin a[i] = OUT_MAX;        w[i] = OUT_MIN;        end
                        4: begin a[i] = OUT_MIN;        w[i] = OUT_MIN;        end
                        default: begin
                            a[i] = DATA_W'($urandom);
                            w[i] = WEIGHT_W'($urandom);
                        end
                    endcase
                    acc += longint'(a[i]) * longint'(w[i]);
                end

                // present this beat
                for (int i = 0; i < 9; i++) begin
                    s_window [i*DATA_W   +: DATA_W  ] = a[i];
                    s_weights[i*WEIGHT_W +: WEIGHT_W] = w[i];
                end
                s_first = (ch == 0);
                s_last  = (ch == nchan-1);

                // handshake, with random gaps
                s_valid = 1;
                forever begin
                    if (stall_in > 0 && ($urandom_range(99) < stall_in)) begin
                        s_valid = 0;
                        @(posedge clk);
                        @(negedge clk);
                        s_valid = 1;
                    end else begin
                        @(posedge clk);
                        if (s_ready) begin @(negedge clk); break; end
                        @(negedge clk);
                    end
                end
                s_valid = 0;
            end

            ref_finish(acc, d, sat);
            exp_data_q.push_back(d);
            exp_sat_q.push_back(sat);
            exp_name_q.push_back(name);
        end
    endtask

    // Consume results and compare.
    initial begin
        longint signed exp_d;
        bit            exp_s;
        string         nm;
        m_ready = 0;
        @(posedge rst_n);
        forever begin
            m_ready = (stall_out == 0) || ($urandom_range(99) >= stall_out);
            @(posedge clk);
            if (m_valid && m_ready) begin
                if (exp_data_q.size() == 0) begin
                    $display("  ERROR: unexpected output %0d", m_data);
                    errors++;
                end else begin
                    exp_d = exp_data_q.pop_front();
                    exp_s = exp_sat_q.pop_front();
                    nm    = exp_name_q.pop_front();
                    if (m_data !== DATA_W'(exp_d) || m_sat !== exp_s) begin
                        errors++;
                        if (errors <= 20)
                            $display("  MISMATCH [%s]: got data=%0d sat=%0b, exp data=%0d sat=%0b",
                                     nm, m_data, m_sat, exp_d, exp_s);
                    end
                    checked++;
                    if (exp_s && exp_d == OUT_MAX) n_sat_hi++;
                    if (exp_s && exp_d == OUT_MIN) n_sat_lo++;
                    if (!exp_s)                    n_exact++;
                    if (m_data != 0)               n_nonzero++;
                end
            end
            @(negedge clk);
        end
    end

    task automatic wait_drain();
        int guard;
        begin
            guard = 0;
            while (exp_data_q.size() > 0 && guard < 100000) begin
                @(posedge clk);
                guard++;
            end
            if (exp_data_q.size() > 0) begin
                $display("  ERROR: %0d results never emitted", exp_data_q.size());
                errors++;
            end
        end
    endtask

    initial begin
        errors  = 0;
        checked = 0;
        stall_in = 0; stall_out = 0;
        s_window = '0; s_weights = '0;
        s_first = 0; s_last = 0; s_tag = 0; s_valid = 0;

        rst_n = 0;
        repeat (3) @(posedge clk);
        rst_n = 1;
        @(negedge clk);

        $display("--- directed ---");
        drive_pixel("zeros",            1, 1);
        drive_pixel("zeros x8",         8, 1);
        drive_pixel("+max x1",          1, 2);
        drive_pixel("+max x8",          8, 2);   // must saturate high
        drive_pixel("mixed sign x1",    1, 3);   // must saturate low
        drive_pixel("mixed sign x8",    8, 3);
        drive_pixel("min*min x1",       1, 4);   // most-negative squared -> positive
        drive_pixel("min*min x128",   128, 4);
        drive_pixel("single channel",   1, 0);
        wait_drain();

        $display("--- random, no stalls ---");
        for (int t = 0; t < 200; t++)
            drive_pixel($sformatf("rand%0d", t), $urandom_range(1, 12), 0);
        wait_drain();

        $display("--- random, both sides stalling ---");
        stall_in = 35; stall_out = 35;
        for (int t = 0; t < 200; t++)
            drive_pixel($sformatf("stall%0d", t), $urandom_range(1, 12), 0);
        wait_drain();

        $display("--- 128-channel runs (conv3/conv4 depth) ---");
        for (int t = 0; t < 10; t++)
            drive_pixel($sformatf("deep%0d", t), 128, 0);
        wait_drain();

        $display("");
        $display("coverage: sat_hi=%0d sat_lo=%0d exact=%0d nonzero=%0d",
                 n_sat_hi, n_sat_lo, n_exact, n_nonzero);

        // Every arm of the round/saturate must have been exercised, or the pass
        // above says nothing about it. Only enforced at the neutral shift: a
        // large OUT_SHIFT legitimately never saturates, and a zero one always
        // does, so demanding both arms there would be a false failure. Other
        // shifts are still checked for correctness, just not for coverage.
        if (OUT_SHIFT == WEIGHT_W - 1) begin
            if (n_sat_hi  == 0) begin $display("  ERROR: high saturation never exercised");     errors++; end
            if (n_sat_lo  == 0) begin $display("  ERROR: low saturation never exercised");      errors++; end
            if (n_exact   == 0) begin $display("  ERROR: non-saturating path never exercised"); errors++; end
        end
        if (n_nonzero == 0) begin $display("  ERROR: every result was zero"); errors++; end

        $display("");
        if (errors == 0) $display("PASS  (%0d pixels checked)", checked);
        else             $display("FAIL  %0d errors, %0d pixels checked", errors, checked);
        $finish;
    end

    initial begin
        #20000000;
        $display("FAIL  timeout, %0d results outstanding", exp_data_q.size());
        $finish;
    end

endmodule
