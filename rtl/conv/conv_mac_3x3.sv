`timescale 1ns / 1ps
//
// conv_mac_3x3
//
// One 3x3 convolution tap: multiplies a 9-pixel window by a 9-weight kernel and
// accumulates the result across input channels, then rounds and saturates back
// to Q1.15.
//
// Channel sequencing is driven by the caller through `first` and `last`. One
// output pixel is produced per (first .. last) run:
//
//     beat 0   first=1 last=0    channel 0
//     beat 1   first=0 last=0    channel 1
//     ...
//     beat N-1 first=0 last=1    channel N-1   -> m_valid, one output pixel
//
// A single-channel layer sets first=1 and last=1 on the same beat.
//
// Fixed point
// -----------
// Activations and weights are both Q1.15: a raw int16 v represents v / 2^15,
// giving the range [-1, +1). A product is therefore Q2.30 in 32 bits, and the
// running sum stays in Q2.30. Converting back to Q1.15 at the end means a
// round and a >>> 15.
//
// Two things here are deliberate, because the superseded design got both wrong:
//
//   1. The accumulator is ACC_W wide with real guard bits, and saturation is
//      tested in that wide domain *before* any narrowing. synapse.v compared
//      `syn_current + mult_ext > MAX_ACC` at the accumulator's own width, so the
//      sum wrapped before the comparison could ever be true and the saturation
//      logic was unreachable. Here the comparison happens on a value that
//      cannot have wrapped.
//
//   2. Units are explicit. syn_current was a raw integer while the LIF's
//      threshold was Q1.15, so the two were never comparable. Everything
//      crossing this module's boundary is Q1.15.
//
// ACC_W sizing: a product is DATA_W+WEIGHT_W bits, and summing 9*C of them needs
// ceil(log2(9*C)) guard bits on top -- 11 for C=128. The default 48 covers
// 32 + 11 with room to spare; assert on it rather than trusting it.
//
module conv_mac_3x3 #(
    parameter int DATA_W    = 16,  // activation width, Q1.15
    parameter int WEIGHT_W  = 16,  // weight width, Q1.15
    parameter int ACC_W     = 48,
    // Right shift applied to the Q2.30 accumulator on the way back to Q1.15.
    // WEIGHT_W-1 is the arithmetically neutral value; each layer will want its
    // own, because a fixed shift only suits one layer's dynamic range. Measured
    // on this block: with uniformly random Q1.15 operands, a 9-tap sum over many
    // channels saturates roughly 65% of the time at the neutral shift. Trained
    // weights are far smaller than full scale, so the real per-layer shift is a
    // property of the trained model and must come from it.
    parameter int OUT_SHIFT = WEIGHT_W - 1
)(
    input  logic                        clk,
    input  logic                        rst_n,

    input  logic [9*DATA_W-1:0]         s_window,    // from line_buffer_3x3
    input  logic [9*WEIGHT_W-1:0]       s_weights,
    input  logic                        s_first,     // first channel of this pixel
    input  logic                        s_last,      // last channel of this pixel
    input  logic                        s_valid,
    output logic                        s_ready,

    output logic signed [DATA_W-1:0]    m_data,      // Q1.15
    output logic                        m_sat,       // result was clamped
    output logic                        m_valid,
    input  logic                        m_ready
);

    localparam int PROD_W = DATA_W + WEIGHT_W;

    localparam logic signed [DATA_W-1:0] OUT_MAX =  (1 <<< (DATA_W-1)) - 1;
    localparam logic signed [DATA_W-1:0] OUT_MIN = -(1 <<< (DATA_W-1));

    // Nine parallel products. Each maps to one DSP48E1: a 16x16 signed multiply
    // fits the 25x18 primitive directly.
    logic signed [PROD_W-1:0] prod [9];
    always_comb begin
        for (int i = 0; i < 9; i++) begin
            prod[i] = $signed(s_window [i*DATA_W   +: DATA_W  ]) *
                      $signed(s_weights[i*WEIGHT_W +: WEIGHT_W]);
        end
    end

    // Sign-extend into the accumulator domain before summing, so the adder tree
    // never truncates.
    logic signed [ACC_W-1:0] dot;
    always_comb begin
        dot = '0;
        for (int i = 0; i < 9; i++) dot += ACC_W'(prod[i]);
    end

    logic signed [ACC_W-1:0] acc;
    logic signed [ACC_W-1:0] acc_next;
    assign acc_next = (s_first ? '0 : acc) + dot;

    // Round-half-up, then saturate. Both happen at ACC_W, before narrowing.
    // OUT_SHIFT == 0 means no scaling, and then there is no rounding bit to add.
    logic signed [ACC_W-1:0] rounded;
    always_comb begin
        if (OUT_SHIFT == 0) rounded = acc_next;
        else                rounded = (acc_next + (ACC_W'(1) <<< (OUT_SHIFT-1))) >>> OUT_SHIFT;
    end

    logic sat_hi, sat_lo;
    assign sat_hi = (rounded > ACC_W'(OUT_MAX));
    assign sat_lo = (rounded < ACC_W'(OUT_MIN));

    logic signed [DATA_W-1:0] clamped;
    always_comb begin
        if      (sat_hi) clamped = OUT_MAX;
        else if (sat_lo) clamped = OUT_MIN;
        else             clamped = DATA_W'(rounded);
    end

    // Hold the pipe while a finished pixel is waiting to be taken, so the
    // accumulator cannot be overwritten before its result is read.
    logic accept;
    assign s_ready = (!m_valid || m_ready);
    assign accept  = s_valid && s_ready;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            acc     <= '0;
            m_data  <= '0;
            m_sat   <= 1'b0;
            m_valid <= 1'b0;
        end else begin
            if (m_valid && m_ready) m_valid <= 1'b0;

            if (accept) begin
                acc <= acc_next;
                if (s_last) begin
                    m_data  <= clamped;
                    m_sat   <= sat_hi | sat_lo;
                    m_valid <= 1'b1;
                end
            end
        end
    end

`ifdef ASSERT_ON
    // Guard bits must cover the worst-case channel count the caller will use.
    initial assert (ACC_W >= DATA_W + WEIGHT_W + 4)
        else $fatal(1, "ACC_W=%0d too narrow for a 9-tap sum", ACC_W);
`endif

endmodule
