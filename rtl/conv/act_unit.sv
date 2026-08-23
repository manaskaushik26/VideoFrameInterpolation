`timescale 1ns / 1ps
//
// act_unit -- activation function, applied across LANES parallel values.
//
// Purely combinational, no handshake of its own: it sits between a producer and
// a consumer without disturbing their valid/ready.
//
// This is the one module the CNN -> SNN swap touches (PROJECT_CONTEXT.md
// section 10). The conv engine, line buffers, weight memory, DMA and host code
// all survive that change unaltered; only MODE changes here, plus an outer
// timestep loop. Keeping the activation a separate, zero-latency module is what
// makes that swap cheap -- do not fold it into conv_mac_3x3.
//
//   MODE 0  passthrough (linear), for the final flow layer which has no
//           activation, only a bias
//   MODE 1  ReLU, the CNN activation, and what to build and train against first
//
// MODE 2 is reserved for LIF. It cannot be implemented here as-is: a spiking
// neuron carries membrane state across timesteps, so it needs clk/rst and a
// timestep enable. When that lands this module gains ports and stops being
// combinational, and every integration testbench must be re-run -- adding a
// pipeline stage on one of two converging paths is what broke composition
// before while all unit tests still passed.
//
module act_unit #(
    parameter int DATA_W = 16,
    parameter int LANES  = 1,
    parameter int MODE   = 1
)(
    input  logic [LANES*DATA_W-1:0] d_in,
    output logic [LANES*DATA_W-1:0] d_out
);

    always_comb begin
        for (int i = 0; i < LANES; i++) begin
            logic signed [DATA_W-1:0] v;
            v = $signed(d_in[i*DATA_W +: DATA_W]);
            case (MODE)
                0:       d_out[i*DATA_W +: DATA_W] = v;
                default: d_out[i*DATA_W +: DATA_W] = (v < 0) ? '0 : v;   // ReLU
            endcase
        end
    end

endmodule
