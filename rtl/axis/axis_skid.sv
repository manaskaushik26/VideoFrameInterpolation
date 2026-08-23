`timescale 1ns / 1ps
//
// axis_skid -- AXI-Stream skid buffer.
//
// Registers both the data path and TVALID, and presents a registered TREADY
// upstream, so a long combinational ready chain does not run the length of the
// design. Full throughput when neither side stalls; one entry of slack absorbs
// the cycle it takes for backpressure to propagate.
//
// This is the block the superseded AXI layer needed and did not have. From
// PROJECT_CONTEXT.md section 6.2, that layer:
//
//   - deasserted TVALID after one cycle regardless of TREADY, losing beats
//   - captured data only when TREADY was already high, dropping on backpressure
//   - asserted TLAST every beat, making every 32-bit word its own packet
//   - had m_axis_tlast_reg removed as unused, so S2MM never saw TLAST and the
//     DMA hung
//   - accepted s_axis_tlast and then dropped it
//   - hardwired s_axis_tready to 1, so the pipeline could never backpressure
//
// The rules this module keeps, and that axis_checker enforces in simulation:
//   - once m_valid is asserted it stays asserted until m_ready is seen
//   - m_data and m_last are stable while m_valid is high and unaccepted
//   - m_valid never depends combinationally on m_ready
//
module axis_skid #(
    parameter int DATA_W = 16
)(
    input  logic                clk,
    input  logic                rst_n,

    input  logic [DATA_W-1:0]   s_data,
    input  logic                s_last,
    input  logic                s_valid,
    output logic                s_ready,

    output logic [DATA_W-1:0]   m_data,
    output logic                m_last,
    output logic                m_valid,
    input  logic                m_ready
);

    logic [DATA_W-1:0] skid_data;
    logic              skid_last;
    logic              skid_valid;

    // Upstream is held off only while the slack entry is occupied. This is a
    // register output, not a function of m_ready, which is the point.
    assign s_ready = !skid_valid;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            m_valid    <= 1'b0;
            m_data     <= '0;
            m_last     <= 1'b0;
            skid_valid <= 1'b0;
            skid_data  <= '0;
            skid_last  <= 1'b0;
        end else begin
            if (!m_valid || m_ready) begin
                // Output register is free: drain the slack entry first so beats
                // cannot be reordered, otherwise take directly from upstream.
                if (skid_valid) begin
                    m_data     <= skid_data;
                    m_last     <= skid_last;
                    m_valid    <= 1'b1;
                    skid_valid <= 1'b0;
                end else begin
                    m_data  <= s_data;
                    m_last  <= s_last;
                    m_valid <= s_valid;
                end
            end else begin
                // Output stalled. Anything accepted this cycle goes to slack;
                // s_ready is already low if it is occupied, so nothing is lost.
                if (s_valid && s_ready) begin
                    skid_data  <= s_data;
                    skid_last  <= s_last;
                    skid_valid <= 1'b1;
                end
            end
        end
    end

endmodule
