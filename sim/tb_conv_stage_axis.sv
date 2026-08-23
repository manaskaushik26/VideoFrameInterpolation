`timescale 1ns / 1ps
//
// Integration testbench for conv_stage_axis: the AXI-Stream-wrapped layer, i.e.
// the unit the DMA actually talks to.
//
// Two things are checked at once:
//
//   1. Correctness -- the output equals a full 3D cross-correlation plus ReLU
//      computed in 64-bit software, same as the other integration tests.
//
//   2. Protocol -- axis_checker monitors both interfaces for AXI-Stream
//      violations, with the expected packet length asserted on each. Every rule
//      it enforces corresponds to a defect that shipped in the superseded
//      design and was invisible to its testbenches because those only ever ran
//      at full rate.
//
// The stalling passes are what make this meaningful. TVALID-withdrawal and
// payload-instability bugs cannot occur when nothing ever stalls, which is
// precisely why the old design passed its own tests and hung on hardware.
//
`ifndef TB_IMG_W
  `define TB_IMG_W 8
`endif
`ifndef TB_IMG_H
  `define TB_IMG_H 6
`endif
`ifndef TB_CH_IN
  `define TB_CH_IN 3
`endif
`ifndef TB_N_OUT
  `define TB_N_OUT 4
`endif
`ifndef TB_SHIFT
  `define TB_SHIFT 20
`endif
`ifndef TB_FRAMES
  `define TB_FRAMES 3
`endif

module tb_conv_stage_axis;

    localparam int DATA_W   = 16;
    localparam int WEIGHT_W = 16;
    localparam int ACC_W    = 48;
    localparam int IMG_W    = `TB_IMG_W;
    localparam int IMG_H    = `TB_IMG_H;
    localparam int CH_IN    = `TB_CH_IN;
    localparam int N_OUT    = `TB_N_OUT;
    localparam int SHIFT    = `TB_SHIFT;
    localparam int NFRAMES  = `TB_FRAMES;
    localparam int NPIX     = IMG_W * IMG_H;
    localparam int IN_BEATS  = NPIX * CH_IN;
    localparam int OUT_BEATS = NPIX * N_OUT;

    localparam int OCW = (N_OUT <= 1) ? 1 : $clog2(N_OUT);
    localparam int ICW = (CH_IN <= 1) ? 1 : $clog2(CH_IN);

    localparam longint signed OUT_MAX =  (1 <<< (DATA_W-1)) - 1;
    localparam longint signed OUT_MIN = -(1 <<< (DATA_W-1));

    logic clk = 0;
    logic rst_n;
    always #5 clk = ~clk;

    logic                  w_we;
    logic [OCW-1:0]        w_out_ch;
    logic [ICW-1:0]        w_in_ch;
    logic [9*WEIGHT_W-1:0] w_data;
    logic                  sat_clear, sat_sticky;

    logic [DATA_W-1:0] s_tdata;
    logic              s_tlast, s_tvalid, s_tready;
    logic [DATA_W-1:0] m_tdata;
    logic              m_tlast, m_tvalid, m_tready;

    conv_stage_axis #(
        .DATA_W (DATA_W), .WEIGHT_W (WEIGHT_W), .IMG_W (IMG_W), .IMG_H (IMG_H),
        .CH_IN (CH_IN), .N_OUT (N_OUT), .ACC_W (ACC_W),
        .OUT_SHIFT (SHIFT), .ACT_MODE (1)
    ) dut (
        .clk (clk), .rst_n (rst_n),
        .w_we (w_we), .w_out_ch (w_out_ch), .w_in_ch (w_in_ch), .w_data (w_data),
        .sat_clear (sat_clear), .sat_sticky (sat_sticky),
        .s_axis_tdata (s_tdata), .s_axis_tlast (s_tlast),
        .s_axis_tvalid (s_tvalid), .s_axis_tready (s_tready),
        .m_axis_tdata (m_tdata), .m_axis_tlast (m_tlast),
        .m_axis_tvalid (m_tvalid), .m_axis_tready (m_tready)
    );

    int in_err, in_beats, in_pkts;
    int out_err, out_beats, out_pkts;

    axis_checker #(.NAME ("S_AXIS"), .DATA_W (DATA_W), .PKT_LEN (IN_BEATS)) u_chk_in (
        .clk (clk), .rst_n (rst_n),
        .tdata (s_tdata), .tlast (s_tlast), .tvalid (s_tvalid), .tready (s_tready),
        .errors (in_err), .beats (in_beats), .packets (in_pkts)
    );

    axis_checker #(.NAME ("M_AXIS"), .DATA_W (DATA_W), .PKT_LEN (OUT_BEATS)) u_chk_out (
        .clk (clk), .rst_n (rst_n),
        .tdata (m_tdata), .tlast (m_tlast), .tvalid (m_tvalid), .tready (m_tready),
        .errors (out_err), .beats (out_beats), .packets (out_pkts)
    );

    logic signed [DATA_W-1:0]   img  [IMG_H][IMG_W][CH_IN];
    logic signed [WEIGHT_W-1:0] kern [N_OUT][CH_IN][3][3];
    logic signed [DATA_W-1:0]   outr [IMG_H][IMG_W][N_OUT];

    int errors, checked, out_idx;
    int stall_in, stall_out;

    function automatic logic signed [DATA_W-1:0] quant_relu(longint signed acc);
        longint signed r;
        begin
            if (SHIFT == 0) r = acc;
            else            r = (acc + (64'sd1 <<< (SHIFT-1))) >>> SHIFT;
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
                        for (int k = 0; k < CH_IN; k++)
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

    task automatic load_weights();
        begin
            for (int n = 0; n < N_OUT; n++)
                for (int k = 0; k < CH_IN; k++) begin
                    for (int r = 0; r < 3; r++)
                        for (int c = 0; c < 3; c++) begin
                            kern[n][k][r][c] = WEIGHT_W'($urandom);
                            w_data[(r*3+c)*WEIGHT_W +: WEIGHT_W] = kern[n][k][r][c];
                        end
                    w_out_ch = OCW'(n); w_in_ch = ICW'(k); w_we = 1'b1;
                    @(posedge clk); @(negedge clk);
                end
            w_we = 1'b0;
        end
    endtask

    task automatic run_frame(int fidx);
        int in_idx, oy, ox, oc;
        logic signed [DATA_W-1:0] exp;
        begin
            in_idx = 0; out_idx = 0;

            fork
                begin : drive
                    // A compliant master: idle cycles are inserted only between
                    // beats, and once a beat is presented TVALID and the payload
                    // are held unchanged until TREADY is seen. Randomising
                    // TVALID every cycle instead would itself violate the
                    // protocol -- axis_checker flags it, which is how this
                    // driver got written correctly.
                    bit pending;
                    pending = 0;
                    while (in_idx < IN_BEATS) begin
                        if (!pending) begin
                            if (stall_in > 0 && ($urandom_range(99) < stall_in)) begin
                                s_tvalid = 0; s_tdata = 'x; s_tlast = 1'b0;
                                @(posedge clk);
                                @(negedge clk);
                                continue;
                            end
                            s_tvalid = 1;
                            s_tdata  = img[(in_idx/CH_IN)/IMG_W][(in_idx/CH_IN)%IMG_W][in_idx%CH_IN];
                            s_tlast  = (in_idx == IN_BEATS-1);
                            pending  = 1;
                        end
                        @(posedge clk);
                        if (s_tready) begin
                            in_idx++;
                            pending = 0;
                        end
                        @(negedge clk);
                    end
                    s_tvalid = 0; s_tdata = 'x; s_tlast = 1'b0;
                end

                begin : collect
                    while (out_idx < OUT_BEATS) begin
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
                                    $display("  MISMATCH f%0d beat %0d (y=%0d,x=%0d,ch=%0d): got %0d exp %0d",
                                             fidx, out_idx, oy, ox, oc, $signed(m_tdata), exp);
                            end
                            if (m_tlast && out_idx != OUT_BEATS-1) begin
                                $display("  ERROR f%0d: TLAST on beat %0d, expected %0d",
                                         fidx, out_idx, OUT_BEATS-1);
                                errors++;
                            end
                            if (!m_tlast && out_idx == OUT_BEATS-1) begin
                                $display("  ERROR f%0d: no TLAST on final beat -- DMA would hang", fidx);
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

    initial begin
        errors = 0; checked = 0;
        stall_in = 0; stall_out = 0;
        s_tvalid = 0; s_tlast = 0; s_tdata = '0;
        m_tready = 0; sat_clear = 0;
        w_we = 0; w_out_ch = '0; w_in_ch = '0; w_data = '0;

        rst_n = 0;
        repeat (3) @(posedge clk);
        rst_n = 1;
        @(negedge clk);

        $display("%0dx%0d  CH_IN=%0d N_OUT=%0d SHIFT=%0d  in_pkt=%0d out_pkt=%0d",
                 IMG_W, IMG_H, CH_IN, N_OUT, SHIFT, IN_BEATS, OUT_BEATS);

        load_weights();

        $display("--- full rate ---");
        for (int f = 0; f < NFRAMES; f++) begin
            for (int y = 0; y < IMG_H; y++)
                for (int x = 0; x < IMG_W; x++)
                    for (int k = 0; k < CH_IN; k++) img[y][x][k] = DATA_W'($urandom);
            compute_reference();
            run_frame(f);
        end

        $display("--- heavy stalls on both sides ---");
        stall_in = 45; stall_out = 45;
        for (int f = 0; f < NFRAMES; f++) begin
            for (int y = 0; y < IMG_H; y++)
                for (int x = 0; x < IMG_W; x++)
                    for (int k = 0; k < CH_IN; k++) img[y][x][k] = DATA_W'($urandom);
            compute_reference();
            run_frame(100 + f);
        end

        $display("");
        $display("protocol: S_AXIS %0d beats / %0d packets / %0d violations",
                 in_beats, in_pkts, in_err);
        $display("protocol: M_AXIS %0d beats / %0d packets / %0d violations",
                 out_beats, out_pkts, out_err);

        if (out_pkts != 2*NFRAMES) begin
            $display("  ERROR: %0d output packets, expected %0d", out_pkts, 2*NFRAMES);
            errors++;
        end
        errors += in_err + out_err;

        $display("");
        if (errors == 0) $display("PASS  (%0d beats, protocol clean)", checked);
        else             $display("FAIL  %0d errors", errors);
        $finish;
    end

    initial begin
        #(IN_BEATS * NFRAMES * 2 * 10 * 100 + 500000);
        $display("FAIL  timeout at out_idx=%0d", out_idx);
        $finish;
    end

endmodule
