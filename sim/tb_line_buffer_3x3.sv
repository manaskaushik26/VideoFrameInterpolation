`timescale 1ns / 1ps
//
// Self-checking testbench for line_buffer_3x3.
//
// Holds the stimulus image in an array, recomputes every expected 3x3 window
// directly from it with explicit bounds checks, and compares tap by tap. The
// reference is written independently of the DUT's addressing scheme so a shared
// off-by-one cannot hide.
//
// Runs three passes over the same image:
//   1. no backpressure, input always valid   -- baseline
//   2. random gaps in s_valid                -- stalls on the input side
//   3. random gaps in both s_valid and m_ready -- stalls on both sides
//
// Pass 3 is the one that matters. A line buffer that only works at full rate is
// the standard way this block goes wrong, and the existing design's AXI-Stream
// layer already has exactly that bug (TVALID deasserting regardless of TREADY).
//
module tb_line_buffer_3x3;

    // Geometry is overridable so one testbench covers every shape:
    //   iverilog -g2012 -DTB_IMG_W=224 -DTB_IMG_H=224 ...
`ifndef TB_IMG_W
  `define TB_IMG_W 7
`endif
`ifndef TB_IMG_H
  `define TB_IMG_H 5
`endif

    localparam int DATA_W = 16;
    localparam int IMG_W  = `TB_IMG_W;
    localparam int IMG_H  = `TB_IMG_H;
    localparam int NPIX   = IMG_W * IMG_H;

    logic                    clk = 0;
    logic                    rst_n;
    logic [DATA_W-1:0]       s_data;
    logic                    s_valid;
    logic                    s_ready;
    logic [9*DATA_W-1:0]     m_window;
    logic                    m_valid;
    logic                    m_ready;
    logic                    m_last;

    always #5 clk = ~clk;

    line_buffer_3x3 #(
        .DATA_W (DATA_W),
        .IMG_W  (IMG_W),
        .IMG_H  (IMG_H)
    ) dut (
        .clk (clk), .rst_n (rst_n),
        .s_data (s_data), .s_valid (s_valid), .s_ready (s_ready),
        .m_window (m_window), .m_valid (m_valid), .m_ready (m_ready), .m_last (m_last)
    );

    // Stimulus image, and the pointers the two sides walk it with.
    logic [DATA_W-1:0] img [IMG_H][IMG_W];
    int in_idx;       // next pixel to drive
    int out_idx;      // next window to check
    int errors;
    int checked;
    int last_seen;

    function automatic logic [DATA_W-1:0] ref_pix(int r, int c);
        if (r < 0 || r >= IMG_H || c < 0 || c >= IMG_W) return '0;
        else                                            return img[r][c];
    endfunction

    // Compare one emitted window against the reference neighbourhood.
    task automatic check_window(int idx);
        int orow, ocol, bit_idx;
        logic [DATA_W-1:0] got, exp;
        begin
            orow = idx / IMG_W;
            ocol = idx % IMG_W;
            for (int dr = -1; dr <= 1; dr++) begin
                for (int dc = -1; dc <= 1; dc++) begin
                    bit_idx = (dr + 1) * 3 + (dc + 1);
                    got = m_window[bit_idx*DATA_W +: DATA_W];
                    exp = ref_pix(orow + dr, ocol + dc);
                    if (got !== exp) begin
                        errors++;
                        if (errors <= 20)
                            $display("  MISMATCH win %0d (r=%0d,c=%0d) tap[%0d][%0d]: got %0d exp %0d",
                                     idx, orow, ocol, dr+1, dc+1, got, exp);
                    end
                end
            end
            checked++;
        end
    endtask

    // One full frame. stall_in / stall_out set how often each side backs off.
    task automatic run_pass(string name, int stall_in, int stall_out);
        begin
            $display("--- pass: %s (stall_in=%0d%% stall_out=%0d%%) ---", name, stall_in, stall_out);
            rst_n   = 0;
            s_valid = 0;
            m_ready = 0;
            s_data  = '0;
            in_idx  = 0;
            out_idx = 0;
            checked = 0;
            last_seen = 0;
            repeat (3) @(posedge clk);
            rst_n = 1;
            @(negedge clk);

            fork
                // input side
                begin
                    while (in_idx < NPIX) begin
                        if (stall_in > 0 && ($urandom_range(99) < stall_in)) begin
                            s_valid = 0;
                            s_data  = 'x;          // must not be sampled while invalid
                        end else begin
                            s_valid = 1;
                            s_data  = img[in_idx / IMG_W][in_idx % IMG_W];
                        end
                        @(posedge clk);
                        if (s_valid && s_ready) in_idx++;
                        @(negedge clk);
                    end
                    s_valid = 0;
                    s_data  = 'x;
                end

                // output side
                begin
                    while (out_idx < NPIX) begin
                        m_ready = (stall_out == 0) || ($urandom_range(99) >= stall_out);
                        @(posedge clk);
                        if (m_valid && m_ready) begin
                            check_window(out_idx);
                            if (m_last) last_seen++;
                            out_idx++;
                        end
                        @(negedge clk);
                    end
                    m_ready = 0;
                end
            join

            if (checked != NPIX)
                $display("  ERROR: emitted %0d windows, expected %0d", checked, NPIX);
            if (last_seen != 1)
                $display("  ERROR: m_last asserted %0d times, expected exactly 1", last_seen);
            if (checked != NPIX || last_seen != 1) errors++;
            $display("  %0d windows checked", checked);
        end
    endtask

    initial begin
        $dumpfile("tb_line_buffer_3x3.vcd");
        $dumpvars(0, tb_line_buffer_3x3);

        errors = 0;
        for (int r = 0; r < IMG_H; r++)
            for (int c = 0; c < IMG_W; c++)
                img[r][c] = DATA_W'(r * IMG_W + c + 1);   // 1..NPIX, so 0 means padding

        run_pass("full rate",      0,  0);
        run_pass("input stalls",  40,  0);
        run_pass("both stall",    40, 40);

        $display("");
        if (errors == 0) $display("PASS  (%0d frames, %0d taps compared)", 3, 3*NPIX*9);
        else             $display("FAIL  %0d errors", errors);
        $finish;
    end

    // Guard against a hang rather than letting the sim spin. Scaled to the frame
    // size, with generous headroom for the stalled passes.
    initial begin
        #(NPIX * 3 * 10 * 20);
        $display("FAIL  timeout: in_idx=%0d out_idx=%0d", in_idx, out_idx);
        $finish;
    end

endmodule
