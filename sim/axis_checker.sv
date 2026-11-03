`timescale 1ns / 1ps
//
// axis_checker -- simulation-only AXI-Stream protocol monitor.
//
// Bind or instantiate this on any stream and it reports violations as they
// happen. Every rule below corresponds to a defect that shipped in the
// superseded design (docs/PROJECT_CONTEXT.md section 6.2) and that its testbenches
// did not catch, because they only ever ran at full rate where none of these
// are observable.
//
// Not synthesisable. Do not add it to rtl/.
//
module axis_checker #(
    parameter string NAME     = "axis",
    parameter int    DATA_W   = 16,
    // Number of beats expected between TLAST pulses. 0 disables the check.
    parameter int    PKT_LEN  = 0
)(
    input logic              clk,
    input logic              rst_n,
    input logic [DATA_W-1:0] tdata,
    input logic              tlast,
    input logic              tvalid,
    input logic              tready,

    output int               errors,
    output int               beats,
    output int               packets
);

    logic              prev_valid;
    logic [DATA_W-1:0] prev_data;
    logic              prev_last;
    logic              prev_ready;
    int                beats_this_pkt;

    // One process drives the counters. Violations are tallied into a local first
    // and added once, so two occurring on the same cycle both get counted --
    // repeated `errors <= errors + 1` in one block would keep only the last.
    always_ff @(posedge clk) begin
        int inc;
        inc = 0;

        if (!rst_n) begin
            prev_valid     <= 1'b0;
            prev_ready     <= 1'b0;
            beats_this_pkt <= 0;
            errors         <= 0;
            beats          <= 0;
            packets        <= 0;
        end else begin
            // Rule 1: TVALID must not be withdrawn before the beat is accepted.
            // This is the "TVALID deasserts after one cycle regardless of
            // TREADY" defect, which silently loses beats.
            if (prev_valid && !prev_ready && !tvalid) begin
                $display("  [%s] AXIS VIOLATION: TVALID withdrawn before TREADY", NAME);
                inc++;
            end

            // Rule 2: payload must be stable while a beat is offered and
            // unaccepted, otherwise the consumer latches a different beat than
            // the one the producer thinks it sent.
            if (prev_valid && !prev_ready && tvalid) begin
                if (tdata !== prev_data) begin
                    $display("  [%s] AXIS VIOLATION: TDATA changed while stalled (%0h -> %0h)",
                             NAME, prev_data, tdata);
                    inc++;
                end
                if (tlast !== prev_last) begin
                    $display("  [%s] AXIS VIOLATION: TLAST changed while stalled", NAME);
                    inc++;
                end
            end

            // Packet accounting. A TLAST on every beat -- the defect that made
            // every word its own packet -- shows up here as a wrong length.
            if (tvalid && tready) begin
                beats <= beats + 1;

                // An X on a qualified beat means undriven logic reaching the
                // interface, which is how the constant-zero design looked from
                // outside.
                if ($isunknown(tdata)) begin
                    $display("  [%s] AXIS VIOLATION: X in TDATA on an accepted beat", NAME);
                    inc++;
                end
                if ($isunknown(tlast)) begin
                    $display("  [%s] AXIS VIOLATION: X in TLAST on an accepted beat", NAME);
                    inc++;
                end

                if (tlast) begin
                    packets        <= packets + 1;
                    beats_this_pkt <= 0;
                    if (PKT_LEN != 0 && (beats_this_pkt + 1) != PKT_LEN) begin
                        $display("  [%s] AXIS VIOLATION: packet of %0d beats, expected %0d",
                                 NAME, beats_this_pkt + 1, PKT_LEN);
                        inc++;
                    end
                end else begin
                    beats_this_pkt <= beats_this_pkt + 1;
                end
            end

            prev_valid <= tvalid;
            prev_ready <= tready;
            prev_data  <= tdata;
            prev_last  <= tlast;

            errors <= errors + inc;
        end
    end

endmodule
