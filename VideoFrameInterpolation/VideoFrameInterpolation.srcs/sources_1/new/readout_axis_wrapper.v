module readout_axis_wrapper (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 clk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF M_AXIS" *)
    input  logic        clk,

    input  logic        rst_n,

    input  logic        out_valid,
    input  logic [7:0]  out_neuron_id,
    input  logic [15:0] out_value,

    // AXI-Stream Master Interface
    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TDATA" *)
    output logic [31:0] m_axis_tdata,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TVALID" *)
    output logic        m_axis_tvalid,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TREADY" *)
    input  logic        m_axis_tready,

    (* X_INTERFACE_INFO = "xilinx.com:interface:axis:1.0 M_AXIS TLAST" *)
    output logic        m_axis_tlast
);

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            m_axis_tvalid <= 1'b0;
            m_axis_tlast  <= 1'b0;
            m_axis_tdata  <= 32'd0;
        end else begin
            if (out_valid && m_axis_tready) begin
                m_axis_tdata  <= {8'd0, out_neuron_id, out_value};
                m_axis_tvalid <= 1'b1;
                m_axis_tlast  <= 1'b1;   // one word = one transfer
            end else begin
                m_axis_tvalid <= 1'b0;
                m_axis_tlast  <= 1'b0;
            end
        end
    end

endmodule
