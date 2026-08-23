`timescale 1ns / 1ps
//
// axis_serializer -- LANES parallel values in, one value per cycle out.
//
// conv_layer completes all N_OUT output channels of a pixel on the same beat.
// Serialising them lane by lane produces exactly the order line_buffer_mc
// expects at its input -- (row, col, channel) with channel minor -- so one
// layer's output feeds the next layer's input with nothing in between. That is
// why the lane order here is channel order and must stay that way.
//
// Combinational passthrough, no storage: m_data is a mux on the input, and the
// input beat is only consumed once its last lane has been accepted. That keeps
// it free of the latency-change hazard that has already bitten this design
// twice, at the cost of holding the producer for LANES cycles -- which is
// exactly right, since the producer needs CH_IN cycles to make the next beat
// anyway.
//
// AXI-Stream rules observed deliberately, since the superseded design violated
// all of them:
//   - m_valid does not depend on m_ready
//   - the presented beat is stable while m_valid is high and not accepted
//   - m_last marks the final lane of the final pixel, once per frame
//
module axis_serializer #(
    parameter int DATA_W = 16,
    parameter int LANES  = 4
)(
    input  logic                    clk,
    input  logic                    rst_n,

    input  logic [LANES*DATA_W-1:0] s_data,
    input  logic                    s_last,
    input  logic                    s_valid,
    output logic                    s_ready,

    output logic [DATA_W-1:0]       m_data,
    output logic                    m_last,
    output logic                    m_valid,
    input  logic                    m_ready
);

    localparam int LW = (LANES <= 1) ? 1 : $clog2(LANES);

    logic [LW-1:0] cnt;
    logic          last_lane;

    assign last_lane = (int'(cnt) == LANES-1);

    assign m_data  = s_data[cnt*DATA_W +: DATA_W];
    assign m_valid = s_valid;
    assign m_last  = s_valid && s_last && last_lane;

    // Release the input beat only when its final lane goes out.
    assign s_ready = m_ready && last_lane;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            cnt <= '0;
        end else if (m_valid && m_ready) begin
            cnt <= last_lane ? '0 : (cnt + 1'b1);
        end
    end

endmodule
