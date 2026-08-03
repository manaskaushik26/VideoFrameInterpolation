module neuron_array #(
    parameter N_NEURONS = 4,
    parameter DATA_W = 16
)(
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 timestep_en,
    input  wire [N_NEURONS-1:0] spike_in,
    output wire [N_NEURONS-1:0] spike_out
);

    genvar i;
    generate
        for (i = 0; i < N_NEURONS; i = i + 1) begin : NEURONS
            lif_neuron u_neuron (
                .clk(clk),
                .rst(rst),
                .timestep_en(timestep_en),

                .valid_in(spike_in[i]),               // ✅ FIX
                .input_current(
                    spike_in[i] ? 16'd16384 : 16'd0   // example current
                ),

                .spike(spike_out[i]),                 // ✅ FIX
                .mem(),                               // unused
                .valid_out()                          // unused
            );
        end
    endgenerate

endmodule
