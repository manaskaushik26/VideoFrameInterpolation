`timescale 1ns / 1ps
//
// line_buffer_mc -- multi-channel 3x3 line buffer
//
// The multi-channel generalisation of line_buffer_3x3. Pixels arrive in raster
// order, channel-minor:
//
//     (row 0, col 0, ch 0..C-1), (row 0, col 1, ch 0..C-1), ...
//
// and it emits one 3x3 window per channel, in the same order, tagged with
// m_first / m_last_ch. That is exactly the stream conv_mac_3x3 consumes, so a
// layer is this block feeding N parallel MACs -- one per output channel -- all
// sharing this single window stream. With 220 DSPs and 9 per MAC that is ~24
// output channels in flight, which is what makes the compute budget work:
// conv3 at 112x112 is 924.8 MMAC, or ~43 ms at 100 MHz across 24 MACs.
//
// Why this is a two-stage pipeline
// --------------------------------
// The single-channel version reads its line stores combinationally, which infers
// LUTRAM. That does not scale: at C=128 the stores are
// 2 * 112 * 128 * 16b = 458,752 bits, which must be BRAM, and BRAM reads are
// registered. So address generation and window assembly are split:
//
//     stage A   counters, address generation, read issue
//     stage B   registered read data arrives, window assembled, output emitted
//
// Both stages advance together on a single `advance`, so there is no independent
// stalling and no skid buffer. The line stores are simple dual-port: read at the
// stage-A address, written at the stage-B address, which are never equal in the
// same cycle.
//
// Column history
// --------------
// A window needs columns ic-2, ic-1, ic of the *same* channel, and consecutive
// columns of one channel are C ticks apart. The two column-delay registers of
// the single-channel version therefore become two C-deep memories per row tap
// (6 total, 6 * 128 * 16b = 12 Kbit, small enough for LUTRAM). They are read
// combinationally and written in the same cycle, which is read-first: the window
// sees the old value, which is the one it wants.
//
// Padding and tick space match line_buffer_3x3: (IMG_H+1) x (IMG_W+1) x C, with
// the extra row and column supplying the zero border and consuming no input.
//
// Channel count is runtime, not a parameter
// ----------------------------------------
// One bitstream must run all six layers, whose input channel counts differ
// (6, 32, 64, 128), so MAX_CH only sizes the memories and cfg_channels selects
// how many are actually used. Spatial size stays a parameter: every layer is
// stride 1 with `same` padding, so IMG_W/IMG_H are identical throughout the
// network.
//
// Addressing uses MAX_CH rather than cfg_channels as the column stride. That
// wastes storage when cfg_channels < MAX_CH, but keeps the address a shift
// instead of a runtime multiply, which would otherwise land on the BRAM
// address path.
//
// cfg_channels must be stable while a frame is in flight; it defines the frame.
//
module line_buffer_mc #(
    parameter int DATA_W   = 16,
    parameter int IMG_W    = 112,
    parameter int IMG_H    = 112,
    parameter int MAX_CH   = 8
)(
    input  logic                    clk,
    input  logic                    rst_n,

    input  logic [$clog2(MAX_CH+1)-1:0] cfg_channels,

    input  logic [DATA_W-1:0]       s_data,
    input  logic                    s_valid,
    output logic                    s_ready,

    output logic [9*DATA_W-1:0]     m_window,
    // Index of the channel this window belongs to. Exposed so a layer can select
    // the matching kernel without keeping a second copy of this counter, which
    // would be free to drift out of step with the window stream.
    output logic [$clog2(MAX_CH > 1 ? MAX_CH : 2)-1:0] m_ch,
    output logic                    m_first,      // channel 0 of this pixel
    output logic                    m_last_ch,    // channel C-1 of this pixel
    output logic                    m_last_frame, // final channel of final pixel
    output logic                    m_valid,
    input  logic                    m_ready
);

    localparam int CW     = $clog2(IMG_W + 1);
    localparam int RW     = $clog2(IMG_H + 1);
    localparam int CHW    = $clog2(MAX_CH > 1 ? MAX_CH : 2);
    localparam int DEPTH  = IMG_W * MAX_CH;
    localparam int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH);

    localparam logic [CW-1:0] C_END = CW'(IMG_W);
    localparam logic [RW-1:0] R_END = RW'(IMG_H);

    // Runtime last-channel index.
    logic [CHW-1:0] ch_end;
    assign ch_end = CHW'(cfg_channels - 1);

    // ---------------- stage A: counters and read issue ----------------
    logic [CHW-1:0] ch;
    logic [CW-1:0]  ic;
    logic [RW-1:0]  ir;

    logic in_image;
    assign in_image = (ir < R_END) && (ic < C_END);

    logic [ADDR_W-1:0] addr;
    assign addr = (ic < C_END) ? ADDR_W'(ic * MAX_CH + ch) : '0;

    // ---------------- stage B: registered read data ----------------
    logic [CHW-1:0]    chq;
    logic [CW-1:0]     icq;
    logic [RW-1:0]     irq;
    logic [ADDR_W-1:0] addrq;
    logic [DATA_W-1:0] live_q;
    logic [DATA_W-1:0] a_q, b_q;
    logic              b_vld;      // stage B holds a tick

    // The two stages must be able to move independently. Tying them to a single
    // `advance` deadlocks at end of frame: the final output's tick is a flush
    // tick needing no input, but stage A has by then wrapped to the next frame
    // and is demanding a pixel that never comes, so the last beat is never
    // emitted. Stage B therefore drains on its own.
    //
    // Offering a beat that cannot retire is equally wrong -- a consumer
    // handshakes it, the stage stays put, and the same beat is presented again,
    // violating AXI-Stream. b_out is the single condition for both.
    logic b_out, b_free, adv_a, b_done;

    assign b_out  = b_vld && (int'(irq) >= 1) && (int'(icq) >= 1);
    assign b_free = !b_vld || !b_out || m_ready;
    assign adv_a  = (!in_image || s_valid) && b_free;

    // A stage-B tick is finished when it is either overwritten or retired; its
    // memory writes happen exactly once, at that point.
    assign b_done = b_vld && (adv_a || (b_out && m_ready));

    assign m_valid      = b_out;
    assign s_ready      = in_image && b_free;
    assign m_ch         = chq;
    assign m_first      = (chq == '0);
    assign m_last_ch    = (chq == ch_end);
    assign m_last_frame = b_out && (irq == R_END) && (icq == C_END) && (chq == ch_end);

    // Line stores. bufB holds row ir-1, bufA row ir-2, both indexed by
    // (column, channel).
    logic [DATA_W-1:0] bufA [DEPTH];
    logic [DATA_W-1:0] bufB [DEPTH];

    // Column history, one C-deep pair per row tap.
    logic [DATA_W-1:0] d1m [3][MAX_CH];
    logic [DATA_W-1:0] d2m [3][MAX_CH];

    logic col_in_image;
    assign col_in_image = (icq < C_END);

    logic [DATA_W-1:0] tap [3];
    always_comb begin
        tap[0] = (int'(irq) >= 2  && col_in_image) ? a_q    : '0;   // row irq-2
        tap[1] = (int'(irq) >= 1  && col_in_image) ? b_q    : '0;   // row irq-1
        tap[2] = (irq < R_END     && col_in_image) ? live_q : '0;   // row irq, live
    end

    always_comb begin
        for (int r = 0; r < 3; r++) begin
            m_window[(r*3 + 2)*DATA_W +: DATA_W] = tap[r];
            m_window[(r*3 + 1)*DATA_W +: DATA_W] = (int'(icq) >= 1) ? d1m[r][chq] : '0;
            m_window[(r*3 + 0)*DATA_W +: DATA_W] = (int'(icq) >= 2) ? d2m[r][chq] : '0;
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            ch <= '0; ic <= '0; ir <= '0;
            chq <= '0; icq <= '0; irq <= '0; addrq <= '0;
            live_q <= '0; a_q <= '0; b_q <= '0;
            b_vld <= 1'b0;
        end else begin
            // -- stage B writes, using the values loaded on a previous cycle.
            // Non-blocking reads below see those old values, which is intended.
            if (b_done) begin
                if ((irq < R_END) && col_in_image) begin
                    bufA[addrq] <= b_q;      // row irq-1 becomes row irq-2
                    bufB[addrq] <= live_q;   // row irq   becomes row irq-1
                end
                for (int r = 0; r < 3; r++) begin
                    d2m[r][chq] <= d1m[r][chq];
                    d1m[r][chq] <= tap[r];
                end
            end

            if (adv_a) begin
                // -- load stage B from stage A, issuing the registered reads
                chq    <= ch;
                icq    <= ic;
                irq    <= ir;
                addrq  <= addr;
                live_q <= s_data;
                a_q    <= bufA[addr];
                b_q    <= bufB[addr];
                b_vld  <= 1'b1;

                // -- advance stage A: channel fastest, then column, then row
                if (ch != ch_end) begin
                    ch <= ch + 1'b1;
                end else begin
                    ch <= '0;
                    if (ic == C_END) begin
                        ic <= '0;
                        ir <= (ir == R_END) ? '0 : (ir + 1'b1);
                    end else begin
                        ic <= ic + 1'b1;
                    end
                end
            end else if (b_out && m_ready) begin
                // retired without a replacement behind it
                b_vld <= 1'b0;
            end
        end
    end

endmodule
