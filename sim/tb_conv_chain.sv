`timescale 1ns / 1ps
//
// Integration testbench: two conv_stage instances chained back to back.
//
//     pixels -> [conv1 -> ReLU -> serialise] -> [conv2 -> ReLU -> serialise] -> pixels
//
// This is the deepest integration test in the project. It checks that a layer's
// output stream is literally consumable as the next layer's input stream, with
// no glue -- format, channel order, handshake and end-of-frame all lining up
// across the boundary.
//
// The reference is a full two-layer forward pass in 64-bit software, and it must
// quantise the intermediate exactly as the hardware does (requantise, then ReLU)
// before feeding layer 2. Comparing only the final output is deliberate: an
// error anywhere -- kernel routing, channel order, the serialiser's lane order,
// a desynchronised end-of-frame -- shows up here, and nothing about the
// intermediate is assumed to be observable.
//
// It also exercises real backpressure through the chain, where stage 2 stalling
// propagates back through stage 1's serialiser into its MAC array and line
// buffer. That path has no equivalent in any single-block test.
//
`ifndef TB_IMG_W
  `define TB_IMG_W 8
`endif
`ifndef TB_IMG_H
  `define TB_IMG_H 6
`endif
`ifndef TB_CH0
  `define TB_CH0 2
`endif
`ifndef TB_CH1
  `define TB_CH1 3
`endif
`ifndef TB_CH2
  `define TB_CH2 2
`endif
`ifndef TB_SHIFT
  `define TB_SHIFT 20
`endif
`ifndef TB_FRAMES
  `define TB_FRAMES 2
`endif

module tb_conv_chain;

    localparam int DATA_W   = 16;
    localparam int WEIGHT_W = 16;
    localparam int ACC_W    = 48;
    localparam int IMG_W    = `TB_IMG_W;
    localparam int IMG_H    = `TB_IMG_H;
    localparam int CH0      = `TB_CH0;
    localparam int CH1      = `TB_CH1;
    localparam int CH2      = `TB_CH2;
    localparam int SHIFT    = `TB_SHIFT;
    localparam int NFRAMES  = `TB_FRAMES;
    localparam int NPIX     = IMG_W * IMG_H;

    localparam int W0 = (CH0 <= 1) ? 1 : $clog2(CH0);
    localparam int W1 = (CH1 <= 1) ? 1 : $clog2(CH1);
    localparam int W2 = (CH2 <= 1) ? 1 : $clog2(CH2);

    localparam longint signed OUT_MAX =  (1 <<< (DATA_W-1)) - 1;
    localparam longint signed OUT_MIN = -(1 <<< (DATA_W-1));

    logic clk = 0;
    logic rst_n;
    always #5 clk = ~clk;

    // stage 1 weight port
    logic                  w1_we;
    logic [W1-1:0]         w1_out;
    logic [W0-1:0]         w1_in;
    logic [9*WEIGHT_W-1:0] w1_data;
    // stage 2 weight port
    logic                  w2_we;
    logic [W2-1:0]         w2_out;
    logic [W1-1:0]         w2_in;
    logic [9*WEIGHT_W-1:0] w2_data;

    logic [DATA_W-1:0] s_data;
    logic              s_valid, s_ready;

    logic [DATA_W-1:0] mid_data;
    logic              mid_last, mid_valid, mid_ready;

    logic [DATA_W-1:0] m_data;
    logic              m_last, m_valid, m_ready;

    logic              sat_clear, s1_sat, s2_sat;

    conv_stage #(
        .DATA_W (DATA_W), .WEIGHT_W (WEIGHT_W), .IMG_W (IMG_W), .IMG_H (IMG_H),
        .MAX_CH (CH0), .N_OUT (CH1), .ACC_W (ACC_W), .ACT_MODE (1)
    ) u_s1 (
        .clk (clk), .rst_n (rst_n),
        .cfg_channels ($clog2(CH0+1)'(CH0)), .cfg_shift (6'(SHIFT)), .cfg_linear (1'b0),
        .w_we (w1_we), .w_out_ch (w1_out), .w_in_ch (w1_in), .w_data (w1_data),
        .s_data (s_data), .s_valid (s_valid), .s_ready (s_ready),
        .m_data (mid_data), .m_last (mid_last), .m_valid (mid_valid), .m_ready (mid_ready),
        .sat_clear (sat_clear), .sat_sticky (s1_sat)
    );

    conv_stage #(
        .DATA_W (DATA_W), .WEIGHT_W (WEIGHT_W), .IMG_W (IMG_W), .IMG_H (IMG_H),
        .MAX_CH (CH1), .N_OUT (CH2), .ACC_W (ACC_W), .ACT_MODE (1)
    ) u_s2 (
        .clk (clk), .rst_n (rst_n),
        .cfg_channels ($clog2(CH1+1)'(CH1)), .cfg_shift (6'(SHIFT)), .cfg_linear (1'b0),
        .w_we (w2_we), .w_out_ch (w2_out), .w_in_ch (w2_in), .w_data (w2_data),
        .s_data (mid_data), .s_valid (mid_valid), .s_ready (mid_ready),
        .m_data (m_data), .m_last (m_last), .m_valid (m_valid), .m_ready (m_ready),
        .sat_clear (sat_clear), .sat_sticky (s2_sat)
    );

    logic signed [DATA_W-1:0]   img  [IMG_H][IMG_W][CH0];
    logic signed [WEIGHT_W-1:0] k1   [CH1][CH0][3][3];
    logic signed [WEIGHT_W-1:0] k2   [CH2][CH1][3][3];
    logic signed [DATA_W-1:0]   mid  [IMG_H][IMG_W][CH1];   // reference intermediate
    logic signed [DATA_W-1:0]   outr [IMG_H][IMG_W][CH2];   // reference output

    int errors, checked, out_idx, n_last;
    int stall_in, stall_out;

    // Requantise exactly as conv_mac_3x3 does, then ReLU as act_unit does.
    function automatic logic signed [DATA_W-1:0] quant_relu(longint signed acc);
        longint signed r;
        begin
            if (SHIFT == 0) r = acc;
            else            r = (acc + (64'sd1 <<< (SHIFT-1))) >>> SHIFT;
            if      (r > OUT_MAX) r = OUT_MAX;
            else if (r < OUT_MIN) r = OUT_MIN;
            if (r < 0) r = 0;                    // ReLU
            return DATA_W'(r);
        end
    endfunction

    // Full two-layer forward pass. Layer 2 consumes the *quantised* layer-1
    // result, so the intermediate must be materialised, not folded.
    task automatic compute_reference();
        longint signed acc;
        int iy, ix;
        begin
            for (int y = 0; y < IMG_H; y++)
                for (int x = 0; x < IMG_W; x++)
                    for (int n = 0; n < CH1; n++) begin
                        acc = 0;
                        for (int k = 0; k < CH0; k++)
                            for (int r = 0; r < 3; r++)
                                for (int c = 0; c < 3; c++) begin
                                    iy = y + r - 1; ix = x + c - 1;
                                    if (iy >= 0 && iy < IMG_H && ix >= 0 && ix < IMG_W)
                                        acc += longint'(img[iy][ix][k]) * longint'(k1[n][k][r][c]);
                                end
                        mid[y][x][n] = quant_relu(acc);
                    end

            for (int y = 0; y < IMG_H; y++)
                for (int x = 0; x < IMG_W; x++)
                    for (int m = 0; m < CH2; m++) begin
                        acc = 0;
                        for (int n = 0; n < CH1; n++)
                            for (int r = 0; r < 3; r++)
                                for (int c = 0; c < 3; c++) begin
                                    iy = y + r - 1; ix = x + c - 1;
                                    if (iy >= 0 && iy < IMG_H && ix >= 0 && ix < IMG_W)
                                        acc += longint'(mid[iy][ix][n]) * longint'(k2[m][n][r][c]);
                                end
                        outr[y][x][m] = quant_relu(acc);
                    end
        end
    endtask

    task automatic load_weights();
        begin
            for (int n = 0; n < CH1; n++)
                for (int k = 0; k < CH0; k++) begin
                    for (int r = 0; r < 3; r++)
                        for (int c = 0; c < 3; c++) begin
                            k1[n][k][r][c] = WEIGHT_W'($urandom);
                            w1_data[(r*3+c)*WEIGHT_W +: WEIGHT_W] = k1[n][k][r][c];
                        end
                    w1_out = W1'(n); w1_in = W0'(k); w1_we = 1'b1;
                    @(posedge clk); @(negedge clk);
                end
            w1_we = 1'b0;

            for (int m = 0; m < CH2; m++)
                for (int n = 0; n < CH1; n++) begin
                    for (int r = 0; r < 3; r++)
                        for (int c = 0; c < 3; c++) begin
                            k2[m][n][r][c] = WEIGHT_W'($urandom);
                            w2_data[(r*3+c)*WEIGHT_W +: WEIGHT_W] = k2[m][n][r][c];
                        end
                    w2_out = W2'(m); w2_in = W1'(n); w2_we = 1'b1;
                    @(posedge clk); @(negedge clk);
                end
            w2_we = 1'b0;
        end
    endtask

    task automatic run_frame(int fidx);
        int in_idx, oy, ox, oc;
        logic signed [DATA_W-1:0] got, exp;
        begin
            in_idx = 0; out_idx = 0; n_last = 0;

            fork
                begin
                    while (in_idx < NPIX*CH0) begin
                        if (stall_in > 0 && ($urandom_range(99) < stall_in)) begin
                            s_valid = 0; s_data = 'x;
                        end else begin
                            s_valid = 1;
                            s_data  = img[(in_idx/CH0)/IMG_W][(in_idx/CH0)%IMG_W][in_idx%CH0];
                        end
                        @(posedge clk);
                        if (s_valid && s_ready) in_idx++;
                        @(negedge clk);
                    end
                    s_valid = 0; s_data = 'x;
                end
                begin
                    while (out_idx < NPIX*CH2) begin
                        m_ready = (stall_out == 0) || ($urandom_range(99) >= stall_out);
                        @(posedge clk);
                        if (m_valid && m_ready) begin
                            oc = out_idx % CH2;
                            ox = (out_idx / CH2) % IMG_W;
                            oy = (out_idx / CH2) / IMG_W;
                            got = m_data;
                            exp = outr[oy][ox][oc];
                            if (got !== exp) begin
                                errors++;
                                if (errors <= 15)
                                    $display("  MISMATCH f%0d beat %0d (y=%0d,x=%0d,ch=%0d): got %0d exp %0d",
                                             fidx, out_idx, oy, ox, oc, got, exp);
                            end
                            if (m_last) begin
                                n_last++;
                                if (out_idx != NPIX*CH2-1) begin
                                    $display("  ERROR f%0d: m_last on beat %0d, expected %0d",
                                             fidx, out_idx, NPIX*CH2-1);
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
        errors = 0; checked = 0;
        stall_in = 0; stall_out = 0;
        s_valid = 0; m_ready = 0; s_data = '0; sat_clear = 0;
        w1_we = 0; w1_out = '0; w1_in = '0; w1_data = '0;
        w2_we = 0; w2_out = '0; w2_in = '0; w2_data = '0;

        rst_n = 0;
        repeat (3) @(posedge clk);
        rst_n = 1;
        @(negedge clk);

        $display("%0dx%0d  %0d -> %0d -> %0d channels, SHIFT=%0d, %0d frames",
                 IMG_W, IMG_H, CH0, CH1, CH2, SHIFT, NFRAMES);

        load_weights();

        $display("--- full rate ---");
        for (int f = 0; f < NFRAMES; f++) begin
            for (int y = 0; y < IMG_H; y++)
                for (int x = 0; x < IMG_W; x++)
                    for (int k = 0; k < CH0; k++) img[y][x][k] = DATA_W'($urandom);
            compute_reference();
            run_frame(f);
        end

        $display("--- both sides stalling ---");
        stall_in = 30; stall_out = 30;
        for (int f = 0; f < NFRAMES; f++) begin
            for (int y = 0; y < IMG_H; y++)
                for (int x = 0; x < IMG_W; x++)
                    for (int k = 0; k < CH0; k++) img[y][x][k] = DATA_W'($urandom);
            compute_reference();
            run_frame(100 + f);
        end

        $display("");
        if (checked != 2*NFRAMES*NPIX*CH2) begin
            $display("  ERROR: checked %0d, expected %0d", checked, 2*NFRAMES*NPIX*CH2);
            errors++;
        end
        $display("saturation sticky: layer1=%0b layer2=%0b", s1_sat, s2_sat);
        if (errors == 0) $display("PASS  (%0d output beats through two chained layers)", checked);
        else             $display("FAIL  %0d errors", errors);
        $finish;
    end

    initial begin
        #(NPIX * (CH0+CH1+CH2) * NFRAMES * 2 * 10 * 80 + 500000);
        $display("FAIL  timeout at out_idx=%0d", out_idx);
        $finish;
    end

endmodule
