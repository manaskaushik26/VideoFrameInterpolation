`timescale 1ns / 1ps
//
// Integration testbench for conv_layer: one line_buffer_mc driving N_OUT
// conv_mac_3x3 instances.
//
// Checks the end-to-end property for every output channel independently against
// a full 3D cross-correlation in 64-bit software. The specific risk this covers
// is kernel routing -- with N_OUT MACs sharing one window stream, a swapped or
// mis-indexed weight lookup produces perfectly plausible output on every channel
// while every channel is wrong. Per-channel comparison against distinct random
// kernels is what catches that, so the kernels are deliberately different across
// output channels.
//
// It also asserts that the MACs' handshakes really do move in lockstep, since
// conv_layer wires instance 0's ready to the line buffer on that assumption.
//
`ifndef TB_IMG_W
  `define TB_IMG_W 12
`endif
`ifndef TB_IMG_H
  `define TB_IMG_H 9
`endif
`ifndef TB_CH_IN
  `define TB_CH_IN 6
`endif
`ifndef TB_N_OUT
  `define TB_N_OUT 4
`endif
`ifndef TB_OUT_SHIFT
  `define TB_OUT_SHIFT 18
`endif
`ifndef TB_FRAMES
  `define TB_FRAMES 2
`endif

module tb_conv_layer;

    localparam int DATA_W    = 16;
    localparam int WEIGHT_W  = 16;
    localparam int ACC_W     = 48;
    localparam int IMG_W     = `TB_IMG_W;
    localparam int IMG_H     = `TB_IMG_H;
    localparam int CH_IN     = `TB_CH_IN;
    localparam int N_OUT     = `TB_N_OUT;
    localparam int OUT_SHIFT = `TB_OUT_SHIFT;
    localparam int NFRAMES   = `TB_FRAMES;
    localparam int NPIX      = IMG_W * IMG_H;
    localparam int NBEATS    = NPIX * CH_IN;

    localparam int OCW = (N_OUT <= 1) ? 1 : $clog2(N_OUT);
    localparam int ICW = (CH_IN <= 1) ? 1 : $clog2(CH_IN);

    localparam longint signed OUT_MAX =  (1 <<< (DATA_W-1)) - 1;
    localparam longint signed OUT_MIN = -(1 <<< (DATA_W-1));

    logic clk = 0;
    logic rst_n;
    always #5 clk = ~clk;

    logic                    w_we;
    logic [OCW-1:0]          w_out_ch;
    logic [ICW-1:0]          w_in_ch;
    logic [9*WEIGHT_W-1:0]   w_data;

    logic [DATA_W-1:0]       s_data;
    logic                    s_valid, s_ready;
    logic [N_OUT*DATA_W-1:0] m_data;
    logic [N_OUT-1:0]        m_sat;
    logic                    m_valid, m_ready, m_last;

    conv_layer #(
        .DATA_W (DATA_W), .WEIGHT_W (WEIGHT_W),
        .IMG_W (IMG_W), .IMG_H (IMG_H),
        .MAX_CH (CH_IN), .N_OUT (N_OUT),
        .ACC_W (ACC_W)
    ) dut (
        .clk (clk), .rst_n (rst_n),
        .cfg_channels ($clog2(CH_IN+1)'(CH_IN)), .cfg_shift (6'(OUT_SHIFT)),
        .w_we (w_we), .w_out_ch (w_out_ch), .w_in_ch (w_in_ch), .w_data (w_data),
        .s_data (s_data), .s_valid (s_valid), .s_ready (s_ready),
        .m_data (m_data), .m_sat (m_sat),
        .m_valid (m_valid), .m_ready (m_ready), .m_last (m_last)
    );

    logic signed [DATA_W-1:0]   img  [IMG_H][IMG_W][CH_IN];
    logic signed [WEIGHT_W-1:0] kern [N_OUT][CH_IN][3][3];

    int errors, checked, out_idx, n_sat, n_last;
    int stall_in, stall_out;

    // All MACs must agree on ready/valid, since conv_layer routes instance 0's
    // ready to the line buffer for all of them.
    always @(posedge clk) begin
        if (rst_n && dut.mac_s_ready !== {N_OUT{dut.mac_s_ready[0]}}) begin
            $display("  ERROR: MAC s_ready diverged: %b", dut.mac_s_ready);
            errors++;
        end
        if (rst_n && dut.mac_m_valid !== {N_OUT{dut.mac_m_valid[0]}}) begin
            $display("  ERROR: MAC m_valid diverged: %b", dut.mac_m_valid);
            errors++;
        end
    end

    function automatic void ref_pixel(int oc, int oy, int ox,
                                      output longint signed d, output bit sat);
        longint signed acc, r;
        int iy, ix;
        begin
            acc = 0;
            for (int k = 0; k < CH_IN; k++)
                for (int kr = 0; kr < 3; kr++)
                    for (int kc = 0; kc < 3; kc++) begin
                        iy = oy + kr - 1;
                        ix = ox + kc - 1;
                        if (iy >= 0 && iy < IMG_H && ix >= 0 && ix < IMG_W)
                            acc += longint'(img[iy][ix][k]) * longint'(kern[oc][k][kr][kc]);
                    end

            if (OUT_SHIFT == 0) r = acc;
            else                r = (acc + (64'sd1 <<< (OUT_SHIFT-1))) >>> OUT_SHIFT;

            if      (r > OUT_MAX) begin d = OUT_MAX; sat = 1; end
            else if (r < OUT_MIN) begin d = OUT_MIN; sat = 1; end
            else                  begin d = r;       sat = 0; end
        end
    endfunction

    // Distinct kernels per output channel, so a mis-routed lookup cannot alias.
    task automatic load_weights();
        begin
            for (int n = 0; n < N_OUT; n++)
                for (int k = 0; k < CH_IN; k++) begin
                    for (int r = 0; r < 3; r++)
                        for (int c = 0; c < 3; c++) begin
                            kern[n][k][r][c] = WEIGHT_W'($urandom);
                            w_data[(r*3 + c)*WEIGHT_W +: WEIGHT_W] = kern[n][k][r][c];
                        end
                    w_out_ch = OCW'(n);
                    w_in_ch  = ICW'(k);
                    w_we     = 1'b1;
                    @(posedge clk);
                    @(negedge clk);
                end
            w_we = 1'b0;
        end
    endtask

    task automatic randomise_image();
        begin
            for (int y = 0; y < IMG_H; y++)
                for (int x = 0; x < IMG_W; x++)
                    for (int k = 0; k < CH_IN; k++)
                        img[y][x][k] = DATA_W'($urandom);
        end
    endtask

    task automatic run_frame(int fidx);
        int in_idx;
        longint signed exp_d;
        bit            exp_s;
        logic signed [DATA_W-1:0] got;
        begin
            in_idx = 0; out_idx = 0; n_last = 0;

            fork
                begin
                    while (in_idx < NBEATS) begin
                        if (stall_in > 0 && ($urandom_range(99) < stall_in)) begin
                            s_valid = 0; s_data = 'x;
                        end else begin
                            s_valid = 1;
                            s_data  = img[(in_idx/CH_IN)/IMG_W][(in_idx/CH_IN)%IMG_W][in_idx%CH_IN];
                        end
                        @(posedge clk);
                        if (s_valid && s_ready) in_idx++;
                        @(negedge clk);
                    end
                    s_valid = 0; s_data = 'x;
                end
                begin
                    while (out_idx < NPIX) begin
                        m_ready = (stall_out == 0) || ($urandom_range(99) >= stall_out);
                        @(posedge clk);
                        if (m_valid && m_ready) begin
                            for (int n = 0; n < N_OUT; n++) begin
                                ref_pixel(n, out_idx / IMG_W, out_idx % IMG_W, exp_d, exp_s);
                                got = m_data[n*DATA_W +: DATA_W];
                                if (got !== DATA_W'(exp_d) || m_sat[n] !== exp_s) begin
                                    errors++;
                                    if (errors <= 15)
                                        $display("  MISMATCH f%0d px %0d oc %0d: got %0d/sat%0b exp %0d/sat%0b",
                                                 fidx, out_idx, n, got, m_sat[n], exp_d, exp_s);
                                end
                                if (exp_s) n_sat++;
                            end
                            if (m_last) begin
                                n_last++;
                                if (out_idx != NPIX-1) begin
                                    $display("  ERROR f%0d: m_last on pixel %0d", fidx, out_idx);
                                    errors++;
                                end
                            end
                            checked++;
                            out_idx++;
                        end
                        @(negedge clk);
                    end
                    m_ready = 0;
                end
            join

            if (n_last != 1) begin
                $display("  ERROR f%0d: m_last asserted %0d times", fidx, n_last);
                errors++;
            end
        end
    endtask

    initial begin
        errors = 0; checked = 0; n_sat = 0;
        stall_in = 0; stall_out = 0;
        s_valid = 0; m_ready = 0; s_data = '0;
        w_we = 0; w_out_ch = '0; w_in_ch = '0; w_data = '0;

        rst_n = 0;
        repeat (3) @(posedge clk);
        rst_n = 1;
        @(negedge clk);

        $display("%0dx%0d, CH_IN=%0d, N_OUT=%0d, OUT_SHIFT=%0d, %0d frames",
                 IMG_W, IMG_H, CH_IN, N_OUT, OUT_SHIFT, NFRAMES);

        load_weights();

        $display("--- full rate ---");
        for (int f = 0; f < NFRAMES; f++) begin randomise_image(); run_frame(f); end

        $display("--- both sides stalling ---");
        stall_in = 35; stall_out = 35;
        for (int f = 0; f < NFRAMES; f++) begin randomise_image(); run_frame(100 + f); end

        $display("");
        $display("coverage: %0d pixels x %0d channels = %0d values, %0d saturated",
                 checked, N_OUT, checked*N_OUT, n_sat);
        if (checked != 2*NFRAMES*NPIX) begin
            $display("  ERROR: checked %0d, expected %0d", checked, 2*NFRAMES*NPIX);
            errors++;
        end

        $display("");
        if (errors == 0) $display("PASS  (%0d output values, %0d MACs)", checked*N_OUT, N_OUT);
        else             $display("FAIL  %0d errors", errors);
        $finish;
    end

    initial begin
        #(NBEATS * NFRAMES * 2 * 10 * 60 + 200000);
        $display("FAIL  timeout at out_idx=%0d", out_idx);
        $finish;
    end

endmodule
