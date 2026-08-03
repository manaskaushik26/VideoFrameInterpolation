module readout #(
    parameter CNT_WIDTH = 8   // output pixel / spike count width
)(
    input  wire                 clk,
    input  wire                 rst,

    input  wire                 spike_in,      // from LIF neuron
    input  wire                 timestep_en,   // valid timestep
    input  wire                 clear_window,  // start new pixel/window
    input  wire                 window_done,   // end of window

    output reg  [CNT_WIDTH-1:0] pixel_value    // readout output
);

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            pixel_value <= 0;
        end
        else if (clear_window) begin
            pixel_value <= 0;
        end
        else if (timestep_en && spike_in) begin
            pixel_value <= pixel_value + 1'b1;
        end
    end

endmodule
