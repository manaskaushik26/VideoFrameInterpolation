module spike_router #(
    parameter SRC_W    = 32,     // source neuron ID width
    parameter DST_W    = 8,     // destination neuron ID width
    parameter WEIGHT_W = 16,    // Q1.15
    parameter FANOUT   = 9,     // 3x3 kernel
    parameter ADDR_W   = 16     // routing table address width
)(
    input  wire                      clk,
    input  wire                      rst,

    // Spike input
    input  wire                      spike_valid,
    input  wire [SRC_W-1:0]          spike_src_id,

    // Routed output (to synapse)
    output logic                     route_valid,
    output logic [DST_W-1:0]         route_dst_id,
    output logic signed [WEIGHT_W-1:0] route_weight,

    // BRAM interface
    output logic [ADDR_W-1:0]        bram_addr,
    input  wire  [DST_W+WEIGHT_W-1:0] bram_dout
);

    // -------------------------------------------------
    // FSM definition (enum)
    // -------------------------------------------------
    typedef enum logic [1:0] {
        IDLE,
        READ,
        OUTPUT
    } fsm_state_t;

    fsm_state_t state, next_state;

    // -------------------------------------------------
    // Internal registers
    // -------------------------------------------------
    logic [SRC_W-1:0] src_id_reg;
    logic [$clog2(FANOUT)-1:0] conn_idx;

    // -------------------------------------------------
    // State register
    // -------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst)
            state <= IDLE;
        else
            state <= next_state;
    end

    // -------------------------------------------------
    // FSM next-state + output logic
    // -------------------------------------------------
    always_comb begin
        // defaults
        next_state   = state;
        route_valid  = 1'b0;
        bram_addr    = '0;

        case (state)

            // =====================================
            // IDLE: wait for spike
            // =====================================
            IDLE: begin
                if (spike_valid)
                    next_state = READ;
            end

            // =====================================
            // READ: issue BRAM address
            // =====================================
            READ: begin
                bram_addr  = (src_id_reg * FANOUT) + conn_idx;
                next_state = OUTPUT;
            end

            // =====================================
            // OUTPUT: emit routed connection
            // =====================================
            OUTPUT: begin
                route_valid = 1'b1;
                next_state  = (conn_idx == FANOUT-1) ? IDLE : READ;
            end

            default: next_state = IDLE;

        endcase
    end

    // -------------------------------------------------
    // Sequential datapath (register updates)
    // -------------------------------------------------
    always_ff @(posedge clk) begin
        if (rst) begin
            conn_idx    <= '0;
            src_id_reg  <= '0;
            route_dst_id <= '0;
            route_weight <= '0;
        end else begin
            case (state)

                IDLE: begin
                    if (spike_valid) begin
                        src_id_reg <= spike_src_id;
                        conn_idx   <= '0;
                    end
                end

                READ: begin
                    // nothing to register here
                end

                OUTPUT: begin
                    route_dst_id <= bram_dout[DST_W+WEIGHT_W-1:WEIGHT_W];
                    route_weight <= bram_dout[WEIGHT_W-1:0];
                    if (conn_idx != FANOUT-1)
                        conn_idx <= conn_idx + 1;
                end

            endcase
        end
    end

endmodule
