`timescale 1ns / 1ps
//
// Integration testbench for vfi_top -- the whole IP, driven the way the PS will
// drive it.
//
// Each layer is the full host sequence:
//
//     write CFG (channels, shift, linear) -> DMA kernels -> DMA feature map in
//                                 -> DMA feature map out -> read STATUS
//
// and every register access goes through the real AXI4-Lite channels rather than
// poking internal signals, because the interface itself is new and the old IP
// had none at all.
//
// What this catches that the layer-level tests cannot:
//   - the AXI-Lite register block actually responding (a slave that never
//     asserts BVALID or RVALID locks the PS bus hard)
//   - configuration written over AXI-Lite reaching the datapath
//   - the ID register, which is how the host confirms it loaded the right
//     bitstream -- the old host pointed at the wrong file and the wrong
//     instance name
//   - the frame counter, which distinguishes "running" from "hung"
//   - CFG[14] `linear`: with it set, negative outputs must come out of the IP
//     unchanged (the flow layer); with it clear again, none may. The bitstream
//     of 2026-08-31 hardwired ReLU and clamped half the flow field to zero.
//
module tb_vfi_top;

    localparam int DATA_W   = 16;
    localparam int WEIGHT_W = 16;
    localparam int IMG_W    = 8;
    localparam int IMG_H    = 6;
    localparam int MAX_CH   = 8;
    localparam int N_OUT    = 4;
    localparam int NPIX     = IMG_W * IMG_H;

    localparam logic [31:0] ID_MAGIC = 32'h5646_4931;

    localparam int A_ID = 'h00, A_CFG = 'h04, A_CTRL = 'h08,
                   A_STATUS = 'h0C, A_FRAMES = 'h10;

    localparam longint signed OUT_MAX =  (1 <<< (DATA_W-1)) - 1;
    localparam longint signed OUT_MIN = -(1 <<< (DATA_W-1));

    logic clk = 0;
    logic rst_n;
    always #5 clk = ~clk;

    logic [4:0]  awaddr;  logic awvalid, awready;
    logic [31:0] wdata;   logic [3:0] wstrb; logic wvalid, wready;
    logic [1:0]  bresp;   logic bvalid, bready;
    logic [4:0]  araddr;  logic arvalid, arready;
    logic [31:0] rdata;   logic [1:0] rresp; logic rvalid, rready;

    logic [WEIGHT_W-1:0] wt_tdata; logic wt_tvalid, wt_tready;
    logic [DATA_W-1:0]   s_tdata;  logic s_tlast, s_tvalid, s_tready;
    logic [DATA_W-1:0]   m_tdata;  logic m_tlast, m_tvalid, m_tready;

    vfi_top #(
        .DATA_W (DATA_W), .WEIGHT_W (WEIGHT_W), .IMG_W (IMG_W), .IMG_H (IMG_H),
        .MAX_CH (MAX_CH), .N_OUT (N_OUT), .ACT_MODE (1)
    ) dut (
        .aclk (clk), .aresetn (rst_n),
        .s_axi_awaddr (awaddr), .s_axi_awvalid (awvalid), .s_axi_awready (awready),
        .s_axi_wdata (wdata), .s_axi_wstrb (wstrb),
        .s_axi_wvalid (wvalid), .s_axi_wready (wready),
        .s_axi_bresp (bresp), .s_axi_bvalid (bvalid), .s_axi_bready (bready),
        .s_axi_araddr (araddr), .s_axi_arvalid (arvalid), .s_axi_arready (arready),
        .s_axi_rdata (rdata), .s_axi_rresp (rresp),
        .s_axi_rvalid (rvalid), .s_axi_rready (rready),
        .s_wt_tdata (wt_tdata), .s_wt_tvalid (wt_tvalid), .s_wt_tready (wt_tready),
        .s_axis_tdata (s_tdata), .s_axis_tlast (s_tlast),
        .s_axis_tvalid (s_tvalid), .s_axis_tready (s_tready),
        .m_axis_tdata (m_tdata), .m_axis_tlast (m_tlast),
        .m_axis_tvalid (m_tvalid), .m_axis_tready (m_tready)
    );

    int out_err, out_beats, out_pkts;
    axis_checker #(.NAME ("M_AXIS"), .DATA_W (DATA_W), .PKT_LEN (NPIX*N_OUT)) u_chk (
        .clk (clk), .rst_n (rst_n),
        .tdata (m_tdata), .tlast (m_tlast), .tvalid (m_tvalid), .tready (m_tready),
        .errors (out_err), .beats (out_beats), .packets (out_pkts)
    );

    logic signed [DATA_W-1:0]   img  [IMG_H][IMG_W][MAX_CH];
    logic signed [WEIGHT_W-1:0] kern [N_OUT][MAX_CH][3][3];
    logic signed [DATA_W-1:0]   outr [IMG_H][IMG_W][N_OUT];

    int errors, checked, out_idx, cur_ch, cur_shift, cur_linear, negatives;
    int stall_in, stall_out;

    // ---------------- AXI4-Lite host tasks ----------------
    task automatic axil_write(int addr, logic [31:0] data);
        begin
            @(negedge clk);
            awaddr = 5'(addr); awvalid = 1'b1;
            wdata  = data;     wvalid  = 1'b1; wstrb = 4'hF;
            bready = 1'b1;
            // Hold until each channel is accepted; they may complete separately.
            fork
                begin while (1) begin @(posedge clk); if (awready) break; end
                      @(negedge clk); awvalid = 1'b0; end
                begin while (1) begin @(posedge clk); if (wready)  break; end
                      @(negedge clk); wvalid  = 1'b0; end
            join
            // A slave that never returns BVALID locks the PS bus; time it out.
            begin : bwait
                int guard;
                guard = 0;
                while (!bvalid && guard < 100) begin @(posedge clk); guard++; end
                if (!bvalid) begin
                    $display("  ERROR: no BVALID for write to 0x%02h -- PS would hang", addr);
                    errors++;
                end
            end
            @(negedge clk); bready = 1'b0;
        end
    endtask

    task automatic axil_read(int addr, output logic [31:0] data);
        int guard;
        begin
            @(negedge clk);
            araddr = 5'(addr); arvalid = 1'b1; rready = 1'b1;
            while (1) begin @(posedge clk); if (arready) break; end
            @(negedge clk); arvalid = 1'b0;
            guard = 0;
            while (!rvalid && guard < 100) begin @(posedge clk); guard++; end
            if (!rvalid) begin
                $display("  ERROR: no RVALID for read of 0x%02h -- PS would hang", addr);
                errors++;
                data = 32'hDEAD_BEEF;
            end else begin
                data = rdata;
            end
            @(negedge clk); rready = 1'b0;
        end
    endtask

    function automatic logic signed [DATA_W-1:0] quant_relu(longint signed acc);
        longint signed r;
        begin
            if (cur_shift == 0) r = acc;
            else                r = (acc + (64'sd1 <<< (cur_shift-1))) >>> cur_shift;
            if      (r > OUT_MAX) r = OUT_MAX;
            else if (r < OUT_MIN) r = OUT_MIN;
            if (r < 0 && !cur_linear) r = 0;
            return DATA_W'(r);
        end
    endfunction

    task automatic compute_reference();
        longint signed acc; int iy, ix;
        begin
            for (int y = 0; y < IMG_H; y++)
                for (int x = 0; x < IMG_W; x++)
                    for (int n = 0; n < N_OUT; n++) begin
                        acc = 0;
                        for (int k = 0; k < cur_ch; k++)
                            for (int r = 0; r < 3; r++)
                                for (int c = 0; c < 3; c++) begin
                                    iy = y + r - 1; ix = x + c - 1;
                                    if (iy >= 0 && iy < IMG_H && ix >= 0 && ix < IMG_W)
                                        acc += longint'(img[iy][ix][k]) * longint'(kern[n][k][r][c]);
                                end
                        outr[y][x][n] = quant_relu(acc);
                    end
        end
    endtask

    task automatic dma_weights();
        begin
            for (int n = 0; n < N_OUT; n++)
                for (int k = 0; k < cur_ch; k++)
                    for (int r = 0; r < 3; r++)
                        for (int c = 0; c < 3; c++) begin
                            kern[n][k][r][c] = WEIGHT_W'($urandom);
                            wt_tdata = kern[n][k][r][c]; wt_tvalid = 1'b1;
                            @(posedge clk); @(negedge clk);
                        end
            wt_tvalid = 1'b0; wt_tdata = 'x;
            @(posedge clk); @(negedge clk);
        end
    endtask

    task automatic dma_frame();
        int in_idx, in_beats_exp, oy, ox, oc;
        bit pending;
        logic signed [DATA_W-1:0] exp;
        begin
            in_idx = 0; out_idx = 0;
            in_beats_exp = NPIX * cur_ch;
            fork
                begin
                    pending = 0;
                    while (in_idx < in_beats_exp) begin
                        if (!pending) begin
                            if (stall_in > 0 && ($urandom_range(99) < stall_in)) begin
                                s_tvalid = 0; s_tdata = 'x; s_tlast = 0;
                                @(posedge clk); @(negedge clk); continue;
                            end
                            s_tvalid = 1;
                            s_tdata  = img[(in_idx/cur_ch)/IMG_W][(in_idx/cur_ch)%IMG_W][in_idx%cur_ch];
                            s_tlast  = (in_idx == in_beats_exp-1);
                            pending  = 1;
                        end
                        @(posedge clk);
                        if (s_tready) begin in_idx++; pending = 0; end
                        @(negedge clk);
                    end
                    s_tvalid = 0; s_tdata = 'x; s_tlast = 0;
                end
                begin
                    while (out_idx < NPIX*N_OUT) begin
                        m_tready = (stall_out == 0) || ($urandom_range(99) >= stall_out);
                        @(posedge clk);
                        if (m_tvalid && m_tready) begin
                            oc = out_idx % N_OUT;
                            ox = (out_idx / N_OUT) % IMG_W;
                            oy = (out_idx / N_OUT) / IMG_W;
                            exp = outr[oy][ox][oc];
                            if ($signed(m_tdata) < 0) negatives++;
                            if (m_tdata !== exp) begin
                                errors++;
                                if (errors <= 10)
                                    $display("  MISMATCH ch=%0d beat %0d: got %0d exp %0d",
                                             cur_ch, out_idx, $signed(m_tdata), exp);
                            end
                            checked++; out_idx++;
                        end
                        @(negedge clk);
                    end
                    m_tready = 0;
                end
            join
        end
    endtask

    // One layer, exactly as the PS will run it.
    task automatic run_layer(int nch, int shift, int lin = 0);
        logic [31:0] v;
        begin
            cur_ch = nch; cur_shift = shift; cur_linear = lin; negatives = 0;

            axil_write(A_CFG, 32'(lin) << 14 | 32'(shift) << 8 | 32'(nch));
            axil_read(A_CFG, v);
            if (v[7:0] !== 8'(nch) || v[13:8] !== 6'(shift) || v[14] !== 1'(lin)) begin
                $display("  ERROR: CFG readback %08h, expected ch=%0d shift=%0d linear=%0d",
                         v, nch, shift, lin);
                errors++;
            end

            axil_write(A_CTRL, 32'h1);      // sat_clear

            dma_weights();
            axil_read(A_STATUS, v);
            if (!v[0]) begin
                $display("  ERROR: weights_loaded not set after DMAing kernels");
                errors++;
            end

            for (int y = 0; y < IMG_H; y++)
                for (int x = 0; x < IMG_W; x++)
                    for (int k = 0; k < nch; k++) img[y][x][k] = DATA_W'($urandom);
            compute_reference();
            dma_frame();

            axil_read(A_STATUS, v);
            $display("  ch=%-2d shift=%-2d %s  status=%02h (loaded=%0b sat=%0b)  %0d negative outputs",
                     nch, shift, lin ? "linear" : "relu  ", v[7:0], v[0], v[1], negatives);
            // The reference above already demands every value match; these two
            // make the point of CFG[14] explicit and fail loudly on their own.
            if (lin && negatives == 0) begin
                $display("  ERROR: linear layer produced no negative outputs -- ReLU still applied?");
                errors++;
            end
            if (!lin && negatives != 0) begin
                $display("  ERROR: ReLU layer produced %0d negative outputs", negatives);
                errors++;
            end
        end
    endtask

    initial begin
        logic [31:0] v;
        errors = 0; checked = 0;
        stall_in = 25; stall_out = 25;
        awvalid=0; wvalid=0; bready=0; arvalid=0; rready=0;
        awaddr='0; wdata='0; wstrb='0; araddr='0;
        wt_tvalid=0; wt_tdata='0;
        s_tvalid=0; s_tlast=0; s_tdata='0; m_tready=0;

        rst_n = 0;
        repeat (4) @(posedge clk);
        rst_n = 1;
        @(negedge clk);

        $display("vfi_top  %0dx%0d  MAX_CH=%0d N_OUT=%0d", IMG_W, IMG_H, MAX_CH, N_OUT);

        // The host's first action: confirm it is talking to the right bitstream.
        axil_read(A_ID, v);
        if (v !== ID_MAGIC) begin
            $display("  ERROR: ID reads %08h, expected %08h", v, ID_MAGIC);
            errors++;
        end else begin
            $display("  ID = %08h  (correct bitstream)", v);
        end

        // Unmapped address must respond, not hang the bus.
        axil_read('h1C, v);

        $display("--- layers, driven over AXI-Lite ---");
        run_layer(1, 20);
        run_layer(4, 22);
        run_layer(MAX_CH, 22);
        // ReLU off, as for the flow layer: negatives must pass through. Two
        // frames back to back, then ReLU back on with no reset in between.
        run_layer(MAX_CH, 22, 1);
        run_layer(4, 20, 1);
        run_layer(2, 15);

        axil_read(A_FRAMES, v);
        $display("");
        $display("FRAMES = %0d (expected 6)", v);
        if (v !== 32'd6) begin
            $display("  ERROR: frame counter wrong");
            errors++;
        end
        $display("protocol: M_AXIS %0d beats / %0d packets / %0d violations",
                 out_beats, out_pkts, out_err);
        errors += out_err;

        $display("");
        if (errors == 0) $display("PASS  (%0d beats, 6 layers via AXI-Lite, 2 linear)", checked);
        else             $display("FAIL  %0d errors", errors);
        $finish;
    end

    initial begin
        #(NPIX * MAX_CH * 6 * 10 * 150 + 3000000);
        $display("FAIL  timeout at out_idx=%0d", out_idx);
        $finish;
    end

endmodule
