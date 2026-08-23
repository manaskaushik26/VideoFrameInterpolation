`timescale 1ns / 1ps
//
// Integration testbench for conv_stage_axis -- the AXI-Stream-wrapped layer the
// DMA talks to.
//
// The central thing this proves is that **one elaborated instance runs several
// different layer shapes**, reconfigured at runtime. That is not a nicety: the
// network has six layers with input channel counts of 6, 32, 64 and 128 and
// different requantisation shifts, and only one of them can be on the device at
// a time. If cfg_channels and cfg_shift were parameters, the design could only
// ever run a single layer.
//
// Each configuration runs the full cycle the host will run:
//   set cfg_channels / cfg_shift -> DMA the weights -> DMA a frame -> check
//
// Checked simultaneously:
//   - correctness, against a 3D cross-correlation plus ReLU in 64-bit software
//   - AXI-Stream protocol, via axis_checker on both interfaces
//
// Stalls are applied on both sides throughout, because none of the protocol
// defects that shipped in the superseded design are observable at full rate --
// which is exactly why its own testbenches passed and the hardware hung.
//
`ifndef TB_IMG_W
  `define TB_IMG_W 8
`endif
`ifndef TB_IMG_H
  `define TB_IMG_H 6
`endif
`ifndef TB_MAX_CH
  `define TB_MAX_CH 8
`endif
`ifndef TB_N_OUT
  `define TB_N_OUT 4
`endif

module tb_conv_stage_axis;

    localparam int DATA_W   = 16;
    localparam int WEIGHT_W = 16;
    localparam int ACC_W    = 48;
    localparam int IMG_W    = `TB_IMG_W;
    localparam int IMG_H    = `TB_IMG_H;
    localparam int MAX_CH   = `TB_MAX_CH;
    localparam int N_OUT    = `TB_N_OUT;
    localparam int NPIX     = IMG_W * IMG_H;
    localparam int CFGW     = $clog2(MAX_CH + 1);

    localparam longint signed OUT_MAX =  (1 <<< (DATA_W-1)) - 1;
    localparam longint signed OUT_MIN = -(1 <<< (DATA_W-1));

    logic clk = 0;
    logic rst_n;
    always #5 clk = ~clk;

    logic [CFGW-1:0]     cfg_channels;
    logic [5:0]          cfg_shift;

    logic [WEIGHT_W-1:0] wt_tdata;
    logic                wt_tvalid, wt_tready, weights_loaded;
    logic                sat_clear, sat_sticky;

    logic [DATA_W-1:0]   s_tdata;
    logic                s_tlast, s_tvalid, s_tready;
    logic [DATA_W-1:0]   m_tdata;
    logic                m_tlast, m_tvalid, m_tready;

    conv_stage_axis #(
        .DATA_W (DATA_W), .WEIGHT_W (WEIGHT_W), .IMG_W (IMG_W), .IMG_H (IMG_H),
        .MAX_CH (MAX_CH), .N_OUT (N_OUT), .ACC_W (ACC_W), .ACT_MODE (1)
    ) dut (
        .clk (clk), .rst_n (rst_n),
        .cfg_channels (cfg_channels), .cfg_shift (cfg_shift),
        .s_wt_tdata (wt_tdata), .s_wt_tvalid (wt_tvalid), .s_wt_tready (wt_tready),
        .weights_loaded (weights_loaded),
        .sat_clear (sat_clear), .sat_sticky (sat_sticky),
        .s_axis_tdata (s_tdata), .s_axis_tlast (s_tlast),
        .s_axis_tvalid (s_tvalid), .s_axis_tready (s_tready),
        .m_axis_tdata (m_tdata), .m_axis_tlast (m_tlast),
        .m_axis_tvalid (m_tvalid), .m_axis_tready (m_tready)
    );

    // PKT_LEN is 0 on the input because its packet length changes with
    // cfg_channels. TLAST position is asserted directly in the collector.
    int in_err, in_beats, in_pkts, out_err, out_beats, out_pkts;

    axis_checker #(.NAME ("S_AXIS"), .DATA_W (DATA_W), .PKT_LEN (0)) u_chk_in (
        .clk (clk), .rst_n (rst_n),
        .tdata (s_tdata), .tlast (s_tlast), .tvalid (s_tvalid), .tready (s_tready),
        .errors (in_err), .beats (in_beats), .packets (in_pkts)
    );

    axis_checker #(.NAME ("M_AXIS"), .DATA_W (DATA_W), .PKT_LEN (NPIX*N_OUT)) u_chk_out (
        .clk (clk), .rst_n (rst_n),
        .tdata (m_tdata), .tlast (m_tlast), .tvalid (m_tvalid), .tready (m_tready),
        .errors (out_err), .beats (out_beats), .packets (out_pkts)
    );

    logic signed [DATA_W-1:0]   img  [IMG_H][IMG_W][MAX_CH];
    logic signed [WEIGHT_W-1:0] kern [N_OUT][MAX_CH][3][3];
    logic signed [DATA_W-1:0]   outr [IMG_H][IMG_W][N_OUT];

    int errors, checked, out_idx;
    int stall_in, stall_out;
    int cur_ch, cur_shift;

    function automatic logic signed [DATA_W-1:0] quant_relu(longint signed acc);
        longint signed r;
        begin
            if (cur_shift == 0) r = acc;
            else                r = (acc + (64'sd1 <<< (cur_shift-1))) >>> cur_shift;
            if      (r > OUT_MAX) r = OUT_MAX;
            else if (r < OUT_MIN) r = OUT_MIN;
            if (r < 0) r = 0;
            return DATA_W'(r);
        end
    endfunction

    task automatic compute_reference();
        longint signed acc;
        int iy, ix;
        begin
            for (int y = 0; y < IMG_H; y++)
                for (int x = 0; x < IMG_W; x++)
                    for (int n = 0; n < N_OUT; n++) begin
                        acc = 0;
                        for (int k = 0; k < cur_ch; k++)
                            for (int r = 0; r < 3; r++)
                                for (int c = 0; c < 3; c++) begin
                                    iy = y + r - 1; ix = x + c - 1;
                                    if (iy >= 0 && iy < IMG_H && ix >= 0 && ix < IMG_W)
                                        acc += longint'(img[iy][ix][k]) * longint'(kern[n][k][r][c]);
                                end
                        outr[y][x][n] = quant_relu(acc);
                    end
        end
    endtask

    // Weights in the flattened (out, in, kh, kw) order of the .bin files, with
    // random gaps to prove the loader does not need a gapless stream.
    task automatic load_weights(int gap_pct);
        int beats;
        begin
            beats = 0;
            for (int n = 0; n < N_OUT; n++)
                for (int k = 0; k < cur_ch; k++)
                    for (int r = 0; r < 3; r++)
                        for (int c = 0; c < 3; c++) begin
                            kern[n][k][r][c] = WEIGHT_W'($urandom);
                            while (gap_pct > 0 && ($urandom_range(99) < gap_pct)) begin
                                wt_tvalid = 1'b0; wt_tdata = 'x;
                                @(posedge clk); @(negedge clk);
                            end
                            wt_tdata  = kern[n][k][r][c];
                            wt_tvalid = 1'b1;
                            @(posedge clk);
                            if (!wt_tready) begin
                                $display("  ERROR: weight loader deasserted tready");
                                errors++;
                            end
                            beats++;
                            @(negedge clk);
                        end
            wt_tvalid = 1'b0; wt_tdata = 'x;
            @(posedge clk); @(negedge clk);

            if (beats != N_OUT*cur_ch*9) begin
                $display("  ERROR: streamed %0d weight beats, expected %0d",
                         beats, N_OUT*cur_ch*9);
                errors++;
            end
            if (!weights_loaded) begin
                $display("  ERROR: weights_loaded not asserted after a full set");
                errors++;
            end
        end
    endtask

    task automatic run_frame();
        int in_idx, in_beats_exp, oy, ox, oc;
        bit pending;
        logic signed [DATA_W-1:0] exp;
        begin
            in_idx = 0; out_idx = 0;
            in_beats_exp = NPIX * cur_ch;

            fork
                begin : drive
                    // Compliant master: idles only between beats, and once a beat
                    // is offered TVALID and payload are held until TREADY.
                    pending = 0;
                    while (in_idx < in_beats_exp) begin
                        if (!pending) begin
                            if (stall_in > 0 && ($urandom_range(99) < stall_in)) begin
                                s_tvalid = 0; s_tdata = 'x; s_tlast = 1'b0;
                                @(posedge clk); @(negedge clk);
                                continue;
                            end
                            s_tvalid = 1;
                            s_tdata  = img[(in_idx/cur_ch)/IMG_W][(in_idx/cur_ch)%IMG_W][in_idx%cur_ch];
                            s_tlast  = (in_idx == in_beats_exp-1);
                            pending  = 1;
                        end
                        @(posedge clk);
                        if (s_tready) begin in_idx++; pending = 0; end
                        @(negedge clk);
                    end
                    s_tvalid = 0; s_tdata = 'x; s_tlast = 1'b0;
                end

                begin : collect
                    while (out_idx < NPIX*N_OUT) begin
                        m_tready = (stall_out == 0) || ($urandom_range(99) >= stall_out);
                        @(posedge clk);
                        if (m_tvalid && m_tready) begin
                            oc = out_idx % N_OUT;
                            ox = (out_idx / N_OUT) % IMG_W;
                            oy = (out_idx / N_OUT) / IMG_W;
                            exp = outr[oy][ox][oc];
                            if (m_tdata !== exp) begin
                                errors++;
                                if (errors <= 15)
                                    $display("  MISMATCH ch=%0d sh=%0d beat %0d (y=%0d,x=%0d,oc=%0d): got %0d exp %0d",
                                             cur_ch, cur_shift, out_idx, oy, ox, oc,
                                             $signed(m_tdata), exp);
                            end
                            if (m_tlast !== (out_idx == NPIX*N_OUT-1)) begin
                                $display("  ERROR: TLAST=%0b on beat %0d of %0d -- DMA would hang or split",
                                         m_tlast, out_idx, NPIX*N_OUT-1);
                                errors++;
                            end
                            checked++;
                            out_idx++;
                        end
                        @(negedge clk);
                    end
                    m_tready = 0;
                end
            join
        end
    endtask

    // One full host-side cycle: configure, load weights, run a frame, check.
    task automatic run_config(int nch, int shift);
        begin
            cur_ch    = nch;
            cur_shift = shift;
            cfg_channels = CFGW'(nch);
            cfg_shift    = 6'(shift);
            @(negedge clk);

            sat_clear = 1'b1; @(posedge clk); @(negedge clk); sat_clear = 1'b0;

            load_weights(25);

            for (int y = 0; y < IMG_H; y++)
                for (int x = 0; x < IMG_W; x++)
                    for (int k = 0; k < nch; k++) img[y][x][k] = DATA_W'($urandom);
            compute_reference();
            run_frame();

            $display("  ch=%-3d shift=%-3d  %0d beats out, sat=%0b",
                     nch, shift, NPIX*N_OUT, sat_sticky);
        end
    endtask

    initial begin
        errors = 0; checked = 0;
        stall_in = 30; stall_out = 30;
        s_tvalid = 0; s_tlast = 0; s_tdata = '0;
        m_tready = 0; sat_clear = 0;
        wt_tvalid = 0; wt_tdata = '0;
        cfg_channels = CFGW'(1); cfg_shift = 6'd15;

        rst_n = 0;
        repeat (3) @(posedge clk);
        rst_n = 1;
        @(negedge clk);

        $display("%0dx%0d  MAX_CH=%0d N_OUT=%0d  (stalls on both sides)",
                 IMG_W, IMG_H, MAX_CH, N_OUT);
        $display("--- one instance, several layer shapes, reconfigured at runtime ---");

        // Distinct channel counts and shifts, as six real layers would need.
        run_config(1,      20);
        run_config(2,      18);
        run_config(MAX_CH, 22);
        run_config(3,      15);
        run_config(MAX_CH, 26);
        run_config(1,      20);   // back to the first, to catch sticky state

        $display("");
        $display("protocol: S_AXIS %0d beats / %0d packets / %0d violations",
                 in_beats, in_pkts, in_err);
        $display("protocol: M_AXIS %0d beats / %0d packets / %0d violations",
                 out_beats, out_pkts, out_err);
        errors += in_err + out_err;

        if (out_pkts != 6) begin
            $display("  ERROR: %0d output packets, expected 6", out_pkts);
            errors++;
        end

        $display("");
        if (errors == 0) $display("PASS  (%0d beats over 6 runtime configurations)", checked);
        else             $display("FAIL  %0d errors", errors);
        $finish;
    end

    initial begin
        #(NPIX * MAX_CH * 6 * 10 * 120 + 2000000);
        $display("FAIL  timeout at out_idx=%0d", out_idx);
        $finish;
    end

endmodule
