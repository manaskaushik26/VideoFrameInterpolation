module synapse_fifo #(
    parameter DATA_W = 24,   // DST_W + WEIGHT_W
    parameter DEPTH  = 16
)(
    input  wire              clk,
    input  wire              rst,

    // Write side (from spike_router)
    input  wire              wr_en,
    input  wire [DATA_W-1:0] din,
    output wire              full,

    // Read side (to synapse accumulator)
    input  wire              rd_en,
    output logic [DATA_W-1:0] dout,
    output wire              empty
);

    localparam ADDR_W = $clog2(DEPTH);

    logic [DATA_W-1:0] mem [0:DEPTH-1];
    logic [ADDR_W:0]   wr_ptr, rd_ptr; // extra bit for full/empty

    // Status
    assign empty = (wr_ptr == rd_ptr);
    assign full  = (wr_ptr[ADDR_W-1:0] == rd_ptr[ADDR_W-1:0]) &&
                   (wr_ptr[ADDR_W]     != rd_ptr[ADDR_W]);

    // Write logic
    always_ff @(posedge clk) begin
        if (rst) begin
            wr_ptr <= '0;
        end else if (wr_en && !full) begin
            mem[wr_ptr[ADDR_W-1:0]] <= din;
            wr_ptr <= wr_ptr + 1'b1;
        end
    end

    // Read logic
    always_ff @(posedge clk) begin
        if (rst) begin
            rd_ptr <= '0;
            dout   <= '0;
        end else if (rd_en && !empty) begin
            dout   <= mem[rd_ptr[ADDR_W-1:0]];
            rd_ptr <= rd_ptr + 1'b1;
        end
    end

endmodule
