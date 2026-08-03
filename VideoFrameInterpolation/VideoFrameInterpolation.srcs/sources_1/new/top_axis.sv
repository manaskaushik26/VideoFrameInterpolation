module top_axis (
    input  logic clk,
    input  logic rst_n,

    // AXI-Stream IN (MM2S)
    input  logic [7:0] s_axis_tdata,
    input  logic       s_axis_tvalid,
    input  logic       s_axis_tlast,
    output logic       s_axis_tready,

    // AXI-Stream OUT (S2MM)
    output logic [31:0] m_axis_tdata,
    output logic        m_axis_tvalid,
    input  logic        m_axis_tready,
    output logic        m_axis_tlast
);

    logic        out_valid;
    logic [7:0]  out_neuron_id;
    logic signed [15:0] out_value;

    // Your existing design
    top u_top (
        .clk           (clk),
        .rst_n         (rst_n),

        .s_axis_tdata  (s_axis_tdata),
        .s_axis_tvalid (s_axis_tvalid),
        .s_axis_tlast  (s_axis_tlast),
        .s_axis_tready (s_axis_tready),

        .out_valid     (out_valid),
        .out_neuron_id (out_neuron_id),
        .out_value     (out_value)
    );

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            m_axis_tvalid <= 1'b0;
            m_axis_tlast  <= 1'b0;
        end else begin
            if (out_valid && m_axis_tready) begin
                m_axis_tdata  <= {8'd0, out_neuron_id, out_value};
                m_axis_tvalid <= 1'b1;
                m_axis_tlast  <= 1'b1; // one word per frame
            end else begin
                m_axis_tvalid <= 1'b0;
                m_axis_tlast  <= 1'b0;
            end
        end
    end

endmodule
