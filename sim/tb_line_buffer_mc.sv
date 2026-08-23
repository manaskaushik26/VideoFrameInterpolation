`timescale 1ns / 1ps
//
// Self-checking testbench for line_buffer_mc.
//
// Same approach as tb_line_buffer_3x3: the expected neighbourhood is recomputed
// from an independent image array with explicit bounds checks, so a shared
// off-by-one cannot hide. Extended for channels -- it also checks the emission
// *order* (channel-minor within each output pixel) and the first/last_ch flags,
// since those are what drive conv_mac_3x3's accumulation and a wrong flag would
// silently corrupt every output pixel.
//
`ifndef TB_IMG_W
  `define TB_IMG_W 7
`endif
`ifndef TB_IMG_H
  `define TB_IMG_H 5
`endif
`ifndef TB_CH
  `define TB_CH 3
`endif

module tb_line_buffer_mc;

    localparam int DATA_W   = 16;
    localparam int IMG_W    = `TB_IMG_W;
    localparam int IMG_H    = `TB_IMG_H;
    localparam int CHANNELS = `TB_CH;
    localparam int NPIX     = IMG_W * IMG_H;
    localparam int NBEATS   = NPIX * CHANNELS;

    logic clk = 0;
    logic rst_n;
    always #5 clk = ~clk;

    logic [DATA_W-1:0]   s_data;
    logic                s_valid, s_ready;
    logic [9*DATA_W-1:0] m_window;
    logic                m_first, m_last_ch, m_last_frame, m_valid, m_ready;

    line_buffer_mc #(
        .DATA_W (DATA_W), .IMG_W (IMG_W), .IMG_H (IMG_H), .MAX_CH (CHANNELS)
    ) dut (
        .clk (clk), .rst_n (rst_n),
        .cfg_channels ($clog2(CHANNELS+1)'(CHANNELS)),
        .s_data (s_data), .s_valid (s_valid), .s_ready (s_ready),
        .m_window (m_window), .m_first (m_first), .m_last_ch (m_last_ch),
        .m_last_frame (m_last_frame), .m_valid (m_valid), .m_ready (m_ready)
    );

    logic [DATA_W-1:0] img [IMG_H][IMG_W][CHANNELS];

    int errors, checked, in_idx, out_idx, n_frame_last;
    int stall_in, stall_out;

    function automatic logic [DATA_W-1:0] ref_pix(int r, int c, int k);
        if (r < 0 || r >= IMG_H || c < 0 || c >= IMG_W) return '0;
        else                                            return img[r][c][k];
    endfunction

    task automatic check_beat(int idx);
        int opix, och, orow, ocol, bidx;
        logic [DATA_W-1:0] got, exp;
        begin
            opix = idx / CHANNELS;
            och  = idx % CHANNELS;
            orow = opix / IMG_W;
            ocol = opix % IMG_W;

            for (int dr = -1; dr <= 1; dr++)
                for (int dc = -1; dc <= 1; dc++) begin
                    bidx = (dr + 1) * 3 + (dc + 1);
                    got  = m_window[bidx*DATA_W +: DATA_W];
                    exp  = ref_pix(orow + dr, ocol + dc, och);
                    if (got !== exp) begin
                        errors++;
                        if (errors <= 20)
                            $display("  MISMATCH beat %0d (r=%0d,c=%0d,ch=%0d) tap[%0d][%0d]: got %0d exp %0d",
                                     idx, orow, ocol, och, dr+1, dc+1, got, exp);
                    end
                end

            // Flags drive the MAC's accumulation window; a wrong one corrupts
            // every output pixel without corrupting any single window.
            if (m_first !== (och == 0)) begin
                errors++;
                if (errors <= 20) $display("  FLAG beat %0d: m_first=%0b, ch=%0d", idx, m_first, och);
            end
            if (m_last_ch !== (och == CHANNELS-1)) begin
                errors++;
                if (errors <= 20) $display("  FLAG beat %0d: m_last_ch=%0b, ch=%0d", idx, m_last_ch, och);
            end
            if (m_last_frame) begin
                n_frame_last++;
                if (idx != NBEATS-1) begin
                    $display("  ERROR: m_last_frame on beat %0d, expected %0d", idx, NBEATS-1);
                    errors++;
                end
            end
            checked++;
        end
    endtask

    task automatic run_pass(string name, int si, int so);
        begin
            $display("--- %s (stall_in=%0d%% stall_out=%0d%%) ---", name, si, so);
            stall_in = si; stall_out = so;
            in_idx = 0; out_idx = 0; checked = 0; n_frame_last = 0;

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
                    while (out_idx < NBEATS) begin
                        m_ready = (stall_out == 0) || ($urandom_range(99) >= stall_out);
                        @(posedge clk);
                        if (m_valid && m_ready) begin
                            check_beat(out_idx);
                            out_idx++;
                        end
                        @(negedge clk);
                    end
                    m_ready = 0;
                end
            join

            if (checked != NBEATS) begin
                $display("  ERROR: %0d beats, expected %0d", checked, NBEATS); errors++;
            end
            if (n_frame_last != 1) begin
                $display("  ERROR: m_last_frame asserted %0d times", n_frame_last); errors++;
            end
            $display("  %0d beats checked", checked);
        end
    endtask

    initial begin
        errors = 0;
        s_valid = 0; m_ready = 0; s_data = '0;

        for (int y = 0; y < IMG_H; y++)
            for (int x = 0; x < IMG_W; x++)
                for (int k = 0; k < CHANNELS; k++)
                    img[y][x][k] = DATA_W'((y*IMG_W + x)*CHANNELS + k + 1);  // 0 means padding

        rst_n = 0;
        repeat (3) @(posedge clk);
        rst_n = 1;
        @(negedge clk);

        $display("geometry %0dx%0d, %0d channels", IMG_W, IMG_H, CHANNELS);
        run_pass("full rate",     0,  0);
        run_pass("input stalls", 40,  0);
        run_pass("both stall",   40, 40);

        $display("");
        if (errors == 0) $display("PASS  (%0d beats/pass, %0d taps compared)", NBEATS, 3*NBEATS*9);
        else             $display("FAIL  %0d errors", errors);
        $finish;
    end

    initial begin
        #(NBEATS * 3 * 10 * 30);
        $display("FAIL  timeout: in=%0d out=%0d", in_idx, out_idx);
        $finish;
    end

endmodule
