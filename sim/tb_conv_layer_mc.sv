`timescale 1ns / 1ps
//
// Integration testbench: line_buffer_mc + conv_mac_3x3 as conv_layer_mc.
//
// Checks only the end-to-end property -- that a whole multi-channel image driven
// through the pair equals a full 3D cross-correlation computed independently in
// 64-bit software, summed over all input channels.
//
// This is the test that matters for the multi-channel path, because the new risk
// is the first/last channel protocol between the two blocks. A wrong flag does
// not corrupt any single window; it corrupts which windows get summed together,
// which is invisible to either unit testbench and fatal to every output pixel.
//
// Frames run back to back with no reset, so a pipeline that only drains cleanly
// once will be caught.
//
`ifndef TB_IMG_W
  `define TB_IMG_W 16
`endif
`ifndef TB_IMG_H
  `define TB_IMG_H 12
`endif
`ifndef TB_CH
  `define TB_CH 6
`endif
`ifndef TB_OUT_SHIFT
  `define TB_OUT_SHIFT 18
`endif
`ifndef TB_FRAMES
  `define TB_FRAMES 2
`endif

module tb_conv_layer_mc;

    localparam int DATA_W    = 16;
    localparam int WEIGHT_W  = 16;
    localparam int ACC_W     = 48;
    localparam int IMG_W     = `TB_IMG_W;
    localparam int IMG_H     = `TB_IMG_H;
    localparam int CHANNELS  = `TB_CH;
    localparam int OUT_SHIFT = `TB_OUT_SHIFT;
    localparam int NFRAMES   = `TB_FRAMES;
    localparam int NPIX      = IMG_W * IMG_H;
    localparam int NBEATS    = NPIX * CHANNELS;

    localparam longint signed OUT_MAX =  (1 <<< (DATA_W-1)) - 1;
    localparam longint signed OUT_MIN = -(1 <<< (DATA_W-1));

    logic clk = 0;
    logic rst_n;
    always #5 clk = ~clk;

    logic [CHANNELS*9*WEIGHT_W-1:0] weights;
    logic [DATA_W-1:0]              s_data;
    logic                           s_valid, s_ready;
    logic signed [DATA_W-1:0]       m_data;
    logic                           m_sat, m_valid, m_ready, m_last;

    conv_layer_mc #(
        .DATA_W (DATA_W), .WEIGHT_W (WEIGHT_W),
        .IMG_W (IMG_W), .IMG_H (IMG_H), .CHANNELS (CHANNELS),
        .ACC_W (ACC_W), .OUT_SHIFT (OUT_SHIFT)
    ) dut (
        .clk (clk), .rst_n (rst_n),
        .weights (weights),
        .s_data (s_data), .s_valid (s_valid), .s_ready (s_ready),
        .m_data (m_data), .m_sat (m_sat),
        .m_valid (m_valid), .m_ready (m_ready), .m_last (m_last)
    );

    logic signed [DATA_W-1:0]   img  [IMG_H][IMG_W][CHANNELS];
    logic signed [WEIGHT_W-1:0] kern [CHANNELS][3][3];

    int errors, checked, out_idx, n_sat, n_last;
    int stall_in, stall_out;

    // Full 3D cross-correlation, `same` zero padding, summed over channels,
    // then the hardware's round-and-saturate. Written from the definition.
    function automatic void ref_pixel(int oy, int ox, output longint signed d, output bit sat);
        longint signed acc, r;
        int iy, ix;
        begin
            acc = 0;
            for (int k = 0; k < CHANNELS; k++)
                for (int kr = 0; kr < 3; kr++)
                    for (int kc = 0; kc < 3; kc++) begin
                        iy = oy + kr - 1;
                        ix = ox + kc - 1;
                        if (iy >= 0 && iy < IMG_H && ix >= 0 && ix < IMG_W)
                            acc += longint'(img[iy][ix][k]) * longint'(kern[k][kr][kc]);
                    end

            if (OUT_SHIFT == 0) r = acc;
            else                r = (acc + (64'sd1 <<< (OUT_SHIFT-1))) >>> OUT_SHIFT;

            if      (r > OUT_MAX) begin d = OUT_MAX; sat = 1; end
            else if (r < OUT_MIN) begin d = OUT_MIN; sat = 1; end
            else                  begin d = r;       sat = 0; end
        end
    endfunction

    task automatic randomise_frame();
        begin
            for (int y = 0; y < IMG_H; y++)
                for (int x = 0; x < IMG_W; x++)
                    for (int k = 0; k < CHANNELS; k++)
                        img[y][x][k] = DATA_W'($urandom);
            for (int k = 0; k < CHANNELS; k++)
                for (int r = 0; r < 3; r++)
                    for (int c = 0; c < 3; c++) begin
                        kern[k][r][c] = WEIGHT_W'($urandom);
                        weights[(k*9 + r*3 + c)*WEIGHT_W +: WEIGHT_W] = kern[k][r][c];
                    end
        end
    endtask

    task automatic run_frame(int fidx);
        int in_idx;
        longint signed exp_d;
        bit            exp_s;
        begin
            in_idx = 0; out_idx = 0; n_last = 0;

            fork
                begin
                    while (in_idx < NBEATS) begin
                        if (stall_in > 0 && ($urandom_range(99) < stall_in)) begin
                            s_valid = 0; s_data = 'x;
                        end else begin
                            s_valid = 1;
                            s_data  = img[(in_idx/CHANNELS)/IMG_W][(in_idx/CHANNELS)%IMG_W][in_idx%CHANNELS];
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
                            ref_pixel(out_idx / IMG_W, out_idx % IMG_W, exp_d, exp_s);
                            if (m_data !== DATA_W'(exp_d) || m_sat !== exp_s) begin
                                errors++;
                                if (errors <= 15)
                                    $display("  MISMATCH f%0d px %0d (y=%0d,x=%0d): got %0d/sat%0b exp %0d/sat%0b",
                                             fidx, out_idx, out_idx/IMG_W, out_idx%IMG_W,
                                             m_data, m_sat, exp_d, exp_s);
                            end
                            if (exp_s) n_sat++;
                            if (m_last) begin
                                n_last++;
                                if (out_idx != NPIX-1) begin
                                    $display("  ERROR f%0d: m_last on pixel %0d, expected %0d",
                                             fidx, out_idx, NPIX-1);
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
                $display("  ERROR f%0d: m_last asserted %0d times, expected 1", fidx, n_last);
                errors++;
            end
        end
    endtask

    initial begin
        errors = 0; checked = 0; n_sat = 0;
        stall_in = 0; stall_out = 0;
        s_valid = 0; m_ready = 0; s_data = '0; weights = '0;

        rst_n = 0;
        repeat (3) @(posedge clk);
        rst_n = 1;
        @(negedge clk);

        $display("geometry %0dx%0d, %0d channels, OUT_SHIFT=%0d, %0d frames",
                 IMG_W, IMG_H, CHANNELS, OUT_SHIFT, NFRAMES);

        $display("--- full rate ---");
        for (int f = 0; f < NFRAMES; f++) begin randomise_frame(); run_frame(f); end

        $display("--- both sides stalling ---");
        stall_in = 35; stall_out = 35;
        for (int f = 0; f < NFRAMES; f++) begin randomise_frame(); run_frame(100 + f); end

        $display("");
        $display("coverage: %0d pixels, %0d saturated", checked, n_sat);
        if (checked != 2*NFRAMES*NPIX) begin
            $display("  ERROR: checked %0d, expected %0d", checked, 2*NFRAMES*NPIX);
            errors++;
        end

        $display("");
        if (errors == 0) $display("PASS  (%0d pixels, %0d channel-beats through both blocks)",
                                  checked, checked*CHANNELS);
        else             $display("FAIL  %0d errors", errors);
        $finish;
    end

    initial begin
        #(NBEATS * NFRAMES * 2 * 10 * 40);
        $display("FAIL  timeout at out_idx=%0d", out_idx);
        $finish;
    end

endmodule
