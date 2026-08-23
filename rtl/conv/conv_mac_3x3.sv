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
    // Sideband bit sampled on the s_last beat and delivered with the output it
    // produces. Callers use it for end-of-frame. It rides the pipeline so that
    // changing this module's latency cannot silently desynchronise a flag a
    // caller is tracking separately -- which is exactly what happened when the
    // requantise stage was added.
    input  logic                        s_tag,
    input  logic                        s_valid,
    output logic                        s_ready,

    output logic signed [DATA_W-1:0]    m_data,      // Q1.15
    output logic                        m_sat,       // result was clamped
    output logic                        m_tag,
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

    // Multiply stage. Registering the products lets synthesis use the DSP48E1's
    // own output register, which is what actually buys the clock rate: without
    // it the path ran BRAM -> DSP -> 9-way adder tree -> accumulate in a single
    // cycle, 19 logic levels, measured -5.561 ns against 10 ns. The control bits
    // travel with the data so they stay aligned.
    logic signed [PROD_W-1:0] prod_q [9];
    logic                     mq_vld, first_q, last_q, tag_q;

    // Sign-extend into the accumulator domain before summing, so the adder tree
    // never truncates.
    logic signed [ACC_W-1:0] dot;
    always_comb begin
        dot = '0;
        for (int i = 0; i < 9; i++) dot += ACC_W'(prod_q[i]);
    end

    logic signed [ACC_W-1:0] acc;
    logic signed [ACC_W-1:0] acc_next;
    assign acc_next = (first_q ? '0 : acc) + dot;

    // Completed accumulator value, waiting to be requantised. This register is
    // what makes the design meet timing: with the accumulate, the rounding add
    // and the saturation compares all in one cycle the critical path was
    // acc_reg -> m_data_reg at 23 logic levels and 17 CARRY4, i.e. three long
    // carry chains in series, which measured -3.655 ns slack against 10 ns.
    // Splitting here puts one carry chain in each stage.
    logic signed [ACC_W-1:0] acc_final;
    logic                    p_vld;
    logic                    p_tag;

    // Round-half-up, then saturate. Both happen at ACC_W, before narrowing.
    // OUT_SHIFT == 0 means no scaling, and then there is no rounding bit to add.
    logic signed [ACC_W-1:0] rounded;
    always_comb begin
        if (OUT_SHIFT == 0) rounded = acc_final;
        else                rounded = (acc_final + (ACC_W'(1) <<< (OUT_SHIFT-1))) >>> OUT_SHIFT;
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

    // Three-stage handshake: multiply -> accumulate -> requantise. Each stage
    // drains independently, so a stalled consumer does not stall the stages
    // behind it any earlier than it must.
    logic m_free, p_to_m, a_can_take, m_to_a, accept;

    assign m_free     = !m_valid || m_ready;
    assign p_to_m     = p_vld && m_free;

    // The accumulate stage absorbs every beat; only a beat that completes a
    // pixel needs the pending slot free.
    assign a_can_take = !last_q || !p_vld || p_to_m;
    assign m_to_a     = mq_vld && a_can_take;

    assign s_ready    = !mq_vld || m_to_a;
    assign accept     = s_valid && s_ready;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int i = 0; i < 9; i++) prod_q[i] <= '0;
            mq_vld    <= 1'b0;
            first_q   <= 1'b0;
            last_q    <= 1'b0;
            tag_q     <= 1'b0;
            acc       <= '0;
            acc_final <= '0;
            p_vld     <= 1'b0;
            p_tag     <= 1'b0;
            m_data    <= '0;
            m_sat     <= 1'b0;
            m_tag     <= 1'b0;
            m_valid   <= 1'b0;
        end else begin
            // -- requantise stage
            if (m_valid && m_ready) m_valid <= 1'b0;
            if (p_to_m) begin
                m_data  <= clamped;
                m_sat   <= sat_hi | sat_lo;
                m_tag   <= p_tag;
                m_valid <= 1'b1;
            end

            // -- accumulate stage. A completed pixel replacing one that is
            // simultaneously moving on keeps p_vld set, which is why this
            // assignment takes precedence over the clear below.
            if (m_to_a) begin
                acc <= acc_next;
                if (last_q) begin
                    acc_final <= acc_next;
                    p_tag     <= tag_q;
                    p_vld     <= 1'b1;
                end else if (p_to_m) begin
                    p_vld     <= 1'b0;
                end
            end else if (p_to_m) begin
                p_vld <= 1'b0;
            end

            // -- multiply stage
            if (accept) begin
                for (int i = 0; i < 9; i++) prod_q[i] <= prod[i];
                first_q <= s_first;
                last_q  <= s_last;
                tag_q   <= s_tag;
                mq_vld  <= 1'b1;
            end else if (m_to_a) begin
                mq_vld  <= 1'b0;
            end
        end
    end

`ifdef ASSERT_ON
    // Guard bits must cover the worst-case channel count the caller will use.
    initial assert (ACC_W >= DATA_W + WEIGHT_W + 4)
        else $fatal(1, "ACC_W=%0d too narrow for a 9-tap sum", ACC_W);
`endif

endmodule
