`timescale 1ns / 1ps
//
// axil_regs -- AXI4-Lite control and status registers.
//
// The superseded IP had no AXI-Lite interface at all: only clk, rst_n, s_axis
// and m_axis. Every ctrl.write() in the host notebook was therefore impossible
// as written (PROJECT_CONTEXT.md section 7). This is that interface.
//
// Register map (byte addresses, 32-bit access)
// -------------------------------------------
//   0x00  ID       R   0x56464931 ("VFI1"). Read it first from the host: the
//                      old host code pointed at the wrong bitstream and the
//                      wrong block-design instance name, and a magic number
//                      turns both of those into an immediate, obvious failure
//                      instead of silently wrong output.
//   0x04  CFG      RW  [7:0]  channels  -- input channels for this layer
//                      [13:8] shift     -- requantisation shift for this layer
//                      Both must be written before the frame that uses them and
//                      held stable for its duration; they define the frame.
//   0x08  CTRL     W   [0] sat_clear  (self-clearing, one-cycle pulse)
//                      [1] soft_reset (self-clearing, one-cycle pulse)
//   0x0C  STATUS   R   [0] weights_loaded
//                      [1] sat_sticky -- any output clipped since sat_clear.
//                          This is how the host tunes the per-layer shift, which
//                          is unknown until the model is trained.
//   0x10  FRAMES   R   completed output frames since reset; counts frame_tick.
//                      Lets the host distinguish "still running" from "hung",
//                      which the old design could not do -- it just hung.
//
// Unmapped addresses read zero and accept writes, deliberately: an AXI-Lite
// slave that never responds locks the PS bus hard, and a wrong address in host
// code is far easier to debug than a frozen board.
//
module axil_regs #(
    parameter int ADDR_W = 5,
    parameter int CH_W   = 8,
    parameter int SH_W   = 6
)(
    input  logic                  clk,
    input  logic                  rst_n,

    // AXI4-Lite slave
    input  logic [ADDR_W-1:0]     s_axil_awaddr,
    input  logic                  s_axil_awvalid,
    output logic                  s_axil_awready,
    input  logic [31:0]           s_axil_wdata,
    input  logic [3:0]            s_axil_wstrb,
    input  logic                  s_axil_wvalid,
    output logic                  s_axil_wready,
    output logic [1:0]            s_axil_bresp,
    output logic                  s_axil_bvalid,
    input  logic                  s_axil_bready,
    input  logic [ADDR_W-1:0]     s_axil_araddr,
    input  logic                  s_axil_arvalid,
    output logic                  s_axil_arready,
    output logic [31:0]           s_axil_rdata,
    output logic [1:0]            s_axil_rresp,
    output logic                  s_axil_rvalid,
    input  logic                  s_axil_rready,

    // control out
    output logic [CH_W-1:0]       cfg_channels,
    output logic [SH_W-1:0]       cfg_shift,
    output logic                  sat_clear,
    output logic                  soft_reset,

    // status in
    input  logic                  weights_loaded,
    input  logic                  sat_sticky,
    input  logic                  frame_tick      // one pulse per completed frame
);

    localparam logic [31:0] ID_MAGIC = 32'h5646_4931;   // "VFI1"

    localparam logic [ADDR_W-1:0] A_ID     = ADDR_W'('h00);
    localparam logic [ADDR_W-1:0] A_CFG    = ADDR_W'('h04);
    localparam logic [ADDR_W-1:0] A_CTRL   = ADDR_W'('h08);
    localparam logic [ADDR_W-1:0] A_STATUS = ADDR_W'('h0C);
    localparam logic [ADDR_W-1:0] A_FRAMES = ADDR_W'('h10);

    logic [31:0] frames;

    // ---------------- write channel ----------------
    // Address and data arrive independently; the write commits once both have
    // been seen, and the response is held until BREADY.
    logic [ADDR_W-1:0] awaddr_q;
    logic [31:0]       wdata_q;
    logic [3:0]        wstrb_q;
    logic              aw_seen, w_seen;

    assign s_axil_awready = !aw_seen && !s_axil_bvalid;
    assign s_axil_wready  = !w_seen  && !s_axil_bvalid;
    assign s_axil_bresp   = 2'b00;    // always OKAY; see header on unmapped
    assign s_axil_rresp   = 2'b00;

    function automatic logic [31:0] wr_merge(logic [31:0] old_v);
        logic [31:0] v;
        begin
            v = old_v;
            for (int b = 0; b < 4; b++)
                if (wstrb_q[b]) v[b*8 +: 8] = wdata_q[b*8 +: 8];
            return v;
        end
    endfunction

    always_ff @(posedge clk) begin
        // Only the CFG fields are kept; the upper bits of the merged word are
        // reserved and intentionally discarded.
        logic [13:0] cfg_new;

        if (!rst_n) begin
            aw_seen       <= 1'b0;
            w_seen        <= 1'b0;
            s_axil_bvalid <= 1'b0;
            cfg_channels  <= CH_W'(1);
            cfg_shift     <= SH_W'(15);
            sat_clear     <= 1'b0;
            soft_reset    <= 1'b0;
        end else begin
            // Control bits are one-cycle pulses.
            sat_clear  <= 1'b0;
            soft_reset <= 1'b0;

            if (s_axil_awvalid && s_axil_awready) begin
                awaddr_q <= s_axil_awaddr;
                aw_seen  <= 1'b1;
            end
            if (s_axil_wvalid && s_axil_wready) begin
                wdata_q <= s_axil_wdata;
                wstrb_q <= s_axil_wstrb;
                w_seen  <= 1'b1;
            end

            if (aw_seen && w_seen && !s_axil_bvalid) begin
                unique case (awaddr_q)
                    A_CFG: begin
                        // Merge under the byte strobes against the current value,
                        // then split the fields back out, so a partial-word write
                        // cannot clobber the field it did not address.
                        cfg_new      = 14'(wr_merge({18'd0, cfg_shift, cfg_channels}));
                        cfg_channels <= cfg_new[CH_W-1:0];
                        cfg_shift    <= cfg_new[8 +: SH_W];
                    end
                    A_CTRL: begin
                        sat_clear  <= wdata_q[0];
                        soft_reset <= wdata_q[1];
                    end
                    default: ;   // unmapped: accepted and ignored
                endcase
                aw_seen       <= 1'b0;
                w_seen        <= 1'b0;
                s_axil_bvalid <= 1'b1;
            end

            if (s_axil_bvalid && s_axil_bready) s_axil_bvalid <= 1'b0;
        end
    end

    // ---------------- read channel ----------------
    assign s_axil_arready = !s_axil_rvalid;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            s_axil_rvalid <= 1'b0;
            s_axil_rdata  <= 32'd0;
        end else begin
            if (s_axil_arvalid && s_axil_arready) begin
                unique case (s_axil_araddr)
                    A_ID:     s_axil_rdata <= ID_MAGIC;
                    A_CFG:    s_axil_rdata <= {18'd0, cfg_shift, cfg_channels};
                    A_STATUS: s_axil_rdata <= {30'd0, sat_sticky, weights_loaded};
                    A_FRAMES: s_axil_rdata <= frames;
                    default:  s_axil_rdata <= 32'd0;
                endcase
                s_axil_rvalid <= 1'b1;
            end
            if (s_axil_rvalid && s_axil_rready) s_axil_rvalid <= 1'b0;
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n)         frames <= 32'd0;
        else if (frame_tick) frames <= frames + 32'd1;
    end

endmodule
