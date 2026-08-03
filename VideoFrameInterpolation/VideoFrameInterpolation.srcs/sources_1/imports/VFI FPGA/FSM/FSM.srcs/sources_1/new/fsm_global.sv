module fsm_global #(
    parameter PIXELS_PER_FRAME = 16
)(
    input  wire clk,
    input  wire rst,
    input  wire start,

    output wire timestep_en,   // now driven by registered signal
    output reg  array_reset,
    output reg  frame_done
);

    // -----------------------------
    // State encoding
    // -----------------------------
    typedef enum logic [2:0] {
        S_RESET,
        S_IDLE,
        S_LOAD,
        S_TIMESTEP,
        S_COLLECT,
        S_NEXT_PIXEL
    } state_t;

    state_t state, next_state;

    reg [$clog2(PIXELS_PER_FRAME)-1:0] pixel_cnt;

    // -----------------------------
    // REGISTERED timestep enable
    // -----------------------------
    reg timestep_en_r;

    assign timestep_en = timestep_en_r;

    // -----------------------------
    // State register
    // -----------------------------
    always @(posedge clk) begin
        if (rst)
            state <= S_RESET;
        else
            state <= next_state;
    end

    // -----------------------------
    // Pixel counter
    // -----------------------------
    always @(posedge clk) begin
        if (rst || state == S_RESET)
            pixel_cnt <= 0;
        else if (state == S_NEXT_PIXEL)
            pixel_cnt <= pixel_cnt + 1;
    end

    // -----------------------------
    // Next-state logic
    // -----------------------------
    always @(*) begin
        next_state = state;
        case (state)
            S_RESET:       next_state = S_IDLE;
            S_IDLE:        if (start) next_state = S_LOAD;
            S_LOAD:        next_state = S_TIMESTEP;
            S_TIMESTEP:    next_state = S_COLLECT;
            S_COLLECT:     next_state = S_NEXT_PIXEL;
            S_NEXT_PIXEL:  next_state =
                            (pixel_cnt == PIXELS_PER_FRAME-1)
                            ? S_IDLE
                            : S_LOAD;
            default:       next_state = S_IDLE;
        endcase
    end

    // -----------------------------
    // REGISTER timestep_en (FIX)
    // -----------------------------
    always @(posedge clk) begin
        if (rst)
            timestep_en_r <= 1'b0;
        else
            timestep_en_r <= (state == S_TIMESTEP);
    end

    // -----------------------------
    // Other outputs (combinational)
    // -----------------------------
    always @(*) begin
        array_reset = 1'b0;
        frame_done  = 1'b0;

        case (state)
            S_RESET: begin
                array_reset = 1'b1;
            end

            S_NEXT_PIXEL: begin
                if (pixel_cnt == PIXELS_PER_FRAME-1)
                    frame_done = 1'b1;
            end
        endcase
    end

endmodule
