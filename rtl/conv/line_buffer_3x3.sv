`timescale 1ns / 1ps
//
// line_buffer_3x3
//
// Streams pixels in raster order and emits the 3x3 neighbourhood of each pixel,
// with `same` zero padding. One output window per input pixel: IMG_H*IMG_W in,
// IMG_H*IMG_W out.
//
// This is the foundation of the convolution engine. It exists so no feature map
// is ever materialised: the widest map in this network is 128*224*224*16b =
// 102.8 Mbit, 21x the whole XC7Z020, while three rows at 128 channels is
// 1.38 Mbit. Everything must stream.
//
// Padding scheme
// --------------
// The window for output pixel (or, oc) covers rows or-1..or+1 and cols oc-1..oc+1,
// with out-of-image taps reading zero. To produce the bottom and right edges the
// pipeline has to run one row and one column past the end of the image, so the
// internal tick space is (IMG_H+1) x (IMG_W+1). During those extra ticks no input
// is consumed -- s_ready deasserts -- and zeros are substituted.
//
// At tick (ir, ic) the emitted window is centred on (ir-1, ic-1), so output is
// valid for ir >= 1 and ic >= 1, giving exactly IMG_H*IMG_W windows.
//
// Window layout
// -------------
// m_window is row-major, flattened, index = row*3 + col:
//
//     [0][0] [0][1] [0][2]      bit  8   7   6
//     [1][0] [1][1] [1][2]  ->       5   4   3     (bit N = m_window[N*DATA_W +: DATA_W])
//     [2][0] [2][1] [2][2]           2   1   0
//
// [1][1] is the centre tap, i.e. the pixel this window belongs to.
//
// Note: the line stores use combinational reads, which infer LUTRAM. At
// IMG_W=224, DATA_W=16 that is 2*224*16 = 7.2 Kbit and is fine. A per-channel
// instance at 128 channels would not be -- that version needs BRAM with
// registered reads and a matching extra pipeline stage.
//
module line_buffer_3x3 #(
    parameter int DATA_W = 16,
    parameter int IMG_W  = 224,
    parameter int IMG_H  = 224
)(
    input  logic                    clk,
    input  logic                    rst_n,

    // pixel stream in, raster order
    input  logic [DATA_W-1:0]       s_data,
    input  logic                    s_valid,
    output logic                    s_ready,

    // 3x3 window out, one per input pixel
    output logic [9*DATA_W-1:0]     m_window,
    output logic                    m_valid,
    input  logic                    m_ready,
    output logic                    m_last
);

    localparam int CW = $clog2(IMG_W + 1);
    localparam int RW = $clog2(IMG_H + 1);

    // Counter-width copies of the bounds, so the comparisons below are not
    // width-expansions -- this project cannot afford to habituate itself to
    // ignoring width warnings. Safe because CW/RW always hold these values:
    // 2**CW >= IMG_W+1 > IMG_W.
    //
    // Note this trick works ONLY for IMG_W/IMG_H. Do not apply it to the small
    // literals 1 and 2 used below: at IMG_W=1 the counter is a single bit, and
    // CW'(2) truncates to 0, turning `ic >= 2` into a tautology. Those
    // comparisons widen the counter with int'() instead. The 1x1 case in the
    // testbench exists to catch exactly this.
    localparam logic [CW-1:0] C_END = CW'(IMG_W);
    localparam logic [RW-1:0] R_END = RW'(IMG_H);

    // Tick counters. Range 0..IMG_W and 0..IMG_H inclusive -- the extra tick per
    // axis is the padding flush described above.
    logic [CW-1:0] ic;
    logic [RW-1:0] ir;

    // Line stores. bufB holds the previous row (ir-1), bufA the one before (ir-2).
    logic [DATA_W-1:0] bufA [IMG_W];
    logic [DATA_W-1:0] bufB [IMG_W];

    // Column delay registers, one pair per row tap.
    logic [DATA_W-1:0] d1 [3];
    logic [DATA_W-1:0] d2 [3];

    // A real pixel is needed only inside the image proper; the flush ticks
    // synthesise zeros instead.
    logic in_image;
    assign in_image = (ir < R_END) && (ic < C_END);

    logic valid_comb;
    assign valid_comb = (int'(ir) >= 1) && (int'(ic) >= 1);

    // Advance when the input we need is present and the output we produce is
    // accepted. Both conditions are vacuous when they do not apply.
    logic advance;
    assign advance = (!in_image || s_valid) && (!valid_comb || m_ready);

    assign s_ready = in_image && (!valid_comb || m_ready);

    // All three taps sit at column ic, so all three are zero once ic runs past
    // the last real column during the flush tick. Clamp the read index as well:
    // ic reaches IMG_W, which is off the end of the line stores.
    logic [CW-1:0] rd_ic;
    assign rd_ic = (ic < C_END) ? ic : '0;

    logic col_in_image;
    assign col_in_image = (ic < C_END);

    logic [DATA_W-1:0] tap [3];
    always_comb begin
        tap[0] = (int'(ir) >= 2 && col_in_image) ? bufA[rd_ic] : '0;   // row ir-2
        tap[1] = (int'(ir) >= 1 && col_in_image) ? bufB[rd_ic] : '0;   // row ir-1
        tap[2] = (ir <  R_END  && col_in_image) ? s_data      : '0;   // row ir, live
    end

    // Window assembly. Column masking handles the left edge, so the delay
    // registers never need clearing between rows.
    always_comb begin
        for (int r = 0; r < 3; r++) begin
            m_window[(r*3 + 2)*DATA_W +: DATA_W] = tap[r];                       // col ic
            m_window[(r*3 + 1)*DATA_W +: DATA_W] = (int'(ic) >= 1) ? d1[r] : '0;       // col ic-1
            m_window[(r*3 + 0)*DATA_W +: DATA_W] = (int'(ic) >= 2) ? d2[r] : '0;       // col ic-2
        end
    end

    // The rightmost column of the window is the live input pixel, so a window is
    // only real once that pixel has actually arrived. Gating m_valid on s_valid
    // is what makes the block correct under input backpressure.
    assign m_valid = valid_comb && (!in_image || s_valid);
    assign m_last  = m_valid && (ir == R_END) && (ic == C_END);

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            ic <= '0;
            ir <= '0;
            for (int r = 0; r < 3; r++) begin
                d1[r] <= '0;
                d2[r] <= '0;
            end
        end else if (advance) begin
            for (int r = 0; r < 3; r++) begin
                d2[r] <= d1[r];
                d1[r] <= tap[r];
            end

            // Shift the row window down one: what was ir-1 becomes ir-2, and the
            // live pixel becomes ir-1. Only meaningful while real pixels arrive.
            if (ir < R_END && ic < C_END) begin
                bufA[ic] <= bufB[ic];
                bufB[ic] <= s_data;
            end

            if (ic == C_END) begin
                ic <= '0;
                ir <= (ir == R_END) ? '0 : (ir + 1'b1);
            end else begin
                ic <= ic + 1'b1;
            end
        end
    end

endmodule
