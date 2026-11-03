`timescale 1ns / 1ps
//
// weight_loader -- streams a layer's kernels in over AXI-Stream and writes them
// into conv_layer's weight memory.
//
// This replaces per-value MMIO writes. The old host notebook (tag legacy-spike-design)
// wrote one weight at a time over AXI-Lite -- 186,624 writes per load, and
// against an interface the IP did not even have (docs/PROJECT_CONTEXT.md section 7).
// A DMA'd blob is one descriptor.
//
// Stream order matches the .bin files exactly
// -------------------------------------------
// Beats arrive as: output channel major, then input channel, then the 9 kernel
// taps in row-major order.
//
//     for n in 0..N_OUT-1:
//       for k in 0..CH_IN-1:
//         w[n][k][0][0], w[n][k][0][1], ... w[n][k][2][2]
//
// which is precisely the flattened (out, in, kh, kw) layout of
// conv*_weight_q15.bin as documented in section 4.3. The host can therefore DMA
// the file as-is, with no reordering -- and any reordering added later would
// silently transpose every kernel, so do not add one.
//
// Total beats for a layer = N_OUT * CH_IN * 9.
//
// TLAST on the stream is not required and not used: the count is determined by
// the parameters. `loaded` rises when a full set has been written and is the
// signal that the layer is safe to run.
//
module weight_loader #(
    parameter int WEIGHT_W = 16,
    parameter int MAX_CH   = 8,
    parameter int N_OUT    = 4
)(
    input  logic                        clk,
    input  logic                        rst_n,

    // How many input channels this layer actually has. The beat count per layer
    // is N_OUT*cfg_channels*9, so this must match what the host DMAs.
    input  logic [$clog2(MAX_CH+1)-1:0] cfg_channels,

    input  logic [WEIGHT_W-1:0]         s_axis_tdata,
    input  logic                        s_axis_tvalid,
    output logic                        s_axis_tready,

    // to conv_layer's weight port
    output logic                        w_we,
    output logic [$clog2(N_OUT > 1 ? N_OUT : 2)-1:0]  w_out_ch,
    output logic [$clog2(MAX_CH > 1 ? MAX_CH : 2)-1:0] w_in_ch,
    output logic [9*WEIGHT_W-1:0]       w_data,

    output logic                        loaded    // full set written
);

    localparam int OCW = $clog2(N_OUT > 1 ? N_OUT : 2);
    localparam int ICW = $clog2(MAX_CH > 1 ? MAX_CH : 2);

    logic [3:0]     tap;      // 0..8
    logic [ICW-1:0] in_ch;
    logic [OCW-1:0] out_ch;

    // Nothing here can back-pressure: each beat is a register write.
    assign s_axis_tready = 1'b1;

    logic accept;
    assign accept = s_axis_tvalid && s_axis_tready;

    logic tap_last;
    assign tap_last = (tap == 4'd8);

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            tap     <= '0;
            in_ch   <= '0;
            out_ch  <= '0;
            w_we    <= 1'b0;
            w_data  <= '0;
            loaded  <= 1'b0;
        end else begin
            w_we <= 1'b0;

            if (accept) begin
                w_data[tap*WEIGHT_W +: WEIGHT_W] <= s_axis_tdata;

                if (tap_last) begin
                    tap      <= '0;
                    // Commit on the next cycle, once w_data holds all nine taps.
                    w_we     <= 1'b1;
                    w_out_ch <= out_ch;
                    w_in_ch  <= in_ch;

                    if (in_ch == ICW'(cfg_channels - 1)) begin
                        in_ch <= '0;
                        if (int'(out_ch) == N_OUT-1) begin
                            out_ch <= '0;
                            loaded <= 1'b1;      // a complete set has landed
                        end else begin
                            out_ch <= out_ch + 1'b1;
                        end
                    end else begin
                        in_ch <= in_ch + 1'b1;
                    end
                end else begin
                    tap <= tap + 1'b1;
                end
            end
        end
    end

endmodule
