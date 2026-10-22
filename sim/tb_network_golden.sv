`timescale 1ns / 1ps
//
// The whole six-layer network on ONE vfi_top, chained, with the trained weights,
// checked value-for-value against the golden model at every layer.
//
// This is the board's job done in simulation: the bitstream's exact
// configuration (112x112, MAX_CH=128, N_OUT=24, ACT_MODE=1), every layer run as
// ceil(c_out/24) passes driven over AXI-Lite and AXI-Stream the way
// host/vfi_driver.run_layer drives it, with stalls on both sides and no reset
// anywhere between passes or layers.
//
// It CHAINS. Each layer's input is the RTL's own stitched output of the layer
// before -- held here in fm[], standing in for the DDR buffer -- never a golden
// file. So an error in any pass propagates to the flow exactly as it would on
// the board, and the per-layer comparison says where it started.
//
//     conv1   6 -> 32   2 passes       conv4 128 -> 64   3 passes
//     conv2  32 -> 64   3 passes       conv5  64 -> 32   2 passes
//     conv3  64 -> 128  6 passes       flow   32 -> 2    1 pass, CFG[14] linear
//
// Inputs come from model/golden_network.py (build/net/); xsim runs in build/:
//
//     python model\golden_network.py
//     .\sim\run_xsim.ps1 -Tb tb_network_golden -Sources <vfi_top rtl list>
//     python model\golden_network.py --check-sim
//
// Every layer's stitched output is also dumped to build/net/sim_<layer>.hex, so
// the final verdict -- including the flow after the host-side bias against
// golden/flow.bin -- is made by Python, independently of this file's checker.
//
module tb_network_golden;

    localparam int DATA_W   = 16;
    localparam int WEIGHT_W = 16;
    localparam int IMG_W    = 112;
    localparam int IMG_H    = 112;
    localparam int MAX_CH   = 128;
    localparam int N_OUT    = 24;
    localparam int N_LAYERS = 6;
    localparam int NPIX     = IMG_W * IMG_H;
    localparam int FM_MAX   = NPIX * MAX_CH;
    localparam int W_MAX    = 6 * N_OUT * MAX_CH * 9;    // conv3/conv4 padded, the largest

    localparam logic [31:0] ID_MAGIC = 32'h5646_4931;
    localparam int A_ID = 'h00, A_CFG = 'h04, A_CTRL = 'h08,
                   A_STATUS = 'h0C, A_FRAMES = 'h10;

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

    // Packet length differs per layer, so the checkers run with PKT_LEN=0
    // (length unchecked) and the TLAST position is asserted below instead.
    int in_err, in_beats, in_pkts;
    axis_checker #(.NAME ("S_AXIS"), .DATA_W (DATA_W), .PKT_LEN (0)) u_chk_in (
        .clk (clk), .rst_n (rst_n),
        .tdata (s_tdata), .tlast (s_tlast), .tvalid (s_tvalid), .tready (s_tready),
        .errors (in_err), .beats (in_beats), .packets (in_pkts)
    );
    int out_err, out_beats, out_pkts;
    axis_checker #(.NAME ("M_AXIS"), .DATA_W (DATA_W), .PKT_LEN (0)) u_chk_out (
        .clk (clk), .rst_n (rst_n),
        .tdata (m_tdata), .tlast (m_tlast), .tvalid (m_tvalid), .tready (m_tready),
        .errors (out_err), .beats (out_beats), .packets (out_pkts)
    );

    // fm is the current layer's input -- the "DDR buffer" the host would hold.
    // nxt collects this layer's passes; it becomes fm for the next layer.
    logic [DATA_W-1:0]   fm  [FM_MAX];
    logic [DATA_W-1:0]   nxt [FM_MAX];
    logic [DATA_W-1:0]   exp [FM_MAX];
    logic [WEIGHT_W-1:0] wts [W_MAX];

    string lname [N_LAYERS];
    int    l_cin [N_LAYERS], l_cout [N_LAYERS], l_shift [N_LAYERS],
           l_lin [N_LAYERS], l_npass [N_LAYERS];
    int    sat_exp [32];

    int errors, layer_err, total_pass, stall_in, stall_out, out_idx, negatives;

    // ---------------- AXI4-Lite host tasks (as tb_vfi_top) ----------------
    task automatic axil_write(int addr, logic [31:0] data);
        int guard;
        begin
            @(negedge clk);
            awaddr = 5'(addr); awvalid = 1'b1;
            wdata  = data;     wvalid  = 1'b1; wstrb = 4'hF;
            bready = 1'b1;
            fork
                begin while (1) begin @(posedge clk); if (awready) break; end
                      @(negedge clk); awvalid = 1'b0; end
                begin while (1) begin @(posedge clk); if (wready)  break; end
                      @(negedge clk); wvalid  = 1'b0; end
            join
            guard = 0;
            while (!bvalid && guard < 100) begin @(posedge clk); guard++; end
            if (!bvalid) begin
                $display("  ERROR: no BVALID for write to 0x%02h -- PS would hang", addr);
                errors++;
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
            end else data = rdata;
            @(negedge clk); rready = 1'b0;
        end
    endtask

    // ---------------- DMA models ----------------
    // Kernels for pass p of the current layer: N_OUT*c_in*9 beats from the
    // padded weight file, offset by pass -- what run_layer slices and sends.
    task automatic dma_weights(int p, int c_in);
        int i, n, base;
        bit pending;
        begin
            n = N_OUT * c_in * 9; base = p * n;
            i = 0; pending = 0;
            while (i < n) begin
                if (!pending) begin
                    if ($urandom_range(99) < stall_in) begin
                        wt_tvalid = 0; wt_tdata = 'x;
                        @(posedge clk); @(negedge clk); continue;
                    end
                    wt_tvalid = 1; wt_tdata = wts[base + i]; pending = 1;
                end
                @(posedge clk);
                if (wt_tready) begin i++; pending = 0; end
                @(negedge clk);
            end
            wt_tvalid = 0; wt_tdata = 'x;
            @(posedge clk); @(negedge clk);
        end
    endtask

    // One pass: fm (c_in channels) in, N_OUT channels out, of which the first
    // `valid` belong to this layer and land in nxt at channel offset p*N_OUT.
    task automatic dma_frame(int l, int p);
        int in_idx, n_in, n_outb, c_in, c_out, pix, j, ch;
        bit pending;
        begin
            c_in = l_cin[l]; c_out = l_cout[l];
            n_in = NPIX * c_in; n_outb = NPIX * N_OUT;
            in_idx = 0; out_idx = 0;
            fork
                begin
                    pending = 0;
                    while (in_idx < n_in) begin
                        if (!pending) begin
                            if ($urandom_range(99) < stall_in) begin
                                s_tvalid = 0; s_tdata = 'x; s_tlast = 0;
                                @(posedge clk); @(negedge clk); continue;
                            end
                            s_tvalid = 1;
                            s_tdata  = fm[in_idx];
                            s_tlast  = (in_idx == n_in - 1);
                            pending  = 1;
                        end
                        @(posedge clk);
                        if (s_tready) begin in_idx++; pending = 0; end
                        @(negedge clk);
                    end
                    s_tvalid = 0; s_tdata = 'x; s_tlast = 0;
                end
                begin
                    while (out_idx < n_outb) begin
                        m_tready = ($urandom_range(99) >= stall_out);
                        @(posedge clk);
                        if (m_tvalid && m_tready) begin
                            pix = out_idx / N_OUT;
                            j   = out_idx % N_OUT;
                            ch  = p * N_OUT + j;
                            if (ch < c_out) begin
                                nxt[pix * c_out + ch] = m_tdata;
                                if ($signed(m_tdata) < 0) negatives++;
                                if (m_tdata !== exp[pix * c_out + ch]) begin
                                    layer_err++;
                                    if (layer_err <= 5)
                                        $display("    MISMATCH %s (row %0d, col %0d, ch %0d): got %0d exp %0d",
                                                 lname[l], pix / IMG_W, pix % IMG_W, ch,
                                                 $signed(m_tdata), $signed(exp[pix * c_out + ch]));
                                end
                            end else if (m_tdata !== '0) begin
                                // A zero-padded kernel must produce exactly zero.
                                layer_err++;
                                if (layer_err <= 5)
                                    $display("    ERROR %s pass %0d: padded channel %0d = %0d, not 0",
                                             lname[l], p, ch, $signed(m_tdata));
                            end
                            if (m_tlast != (out_idx == n_outb - 1)) begin
                                errors++;
                                $display("  ERROR: TLAST=%0b on beat %0d of %0d", m_tlast, out_idx, n_outb);
                            end
                            out_idx++;
                        end
                        @(negedge clk);
                    end
                    m_tready = 0;
                end
            join
        end
    endtask

    task automatic run_layer(int l);
        logic [31:0] cfg, v;
        int fd;
        begin
            $readmemh($sformatf("net/w_%s.hex", lname[l]), wts);
            $readmemh($sformatf("net/exp_%s.hex", lname[l]), exp);
            layer_err = 0; negatives = 0;

            cfg = 32'(l_lin[l]) << 14 | 32'(l_shift[l]) << 8 | 32'(l_cin[l]);
            for (int p = 0; p < l_npass[l]; p++) begin
                // run_layer writes CFG and clears saturation for every pass.
                axil_write(A_CFG, cfg);
                axil_read(A_CFG, v);
                if (v[14:0] !== cfg[14:0]) begin
                    $display("  ERROR: CFG readback %08h, wrote %08h", v, cfg); errors++;
                end
                axil_write(A_CTRL, 32'h1);              // sat_clear

                dma_weights(p, l_cin[l]);
                axil_read(A_STATUS, v);
                if (!v[0]) begin
                    $display("  ERROR: %s pass %0d: weights_loaded not set", lname[l], p); errors++;
                end

                dma_frame(l, p);

                axil_read(A_STATUS, v);
                if (v[1] !== sat_exp[total_pass][0]) begin
                    $display("  ERROR: %s pass %0d: sat_sticky %0b, golden %0d",
                             lname[l], p, v[1], sat_exp[total_pass]); errors++;
                end
                total_pass++;
            end

            // Dump the stitched layer, then make it the next layer's input.
            fd = $fopen($sformatf("net/sim_%s.hex", lname[l]), "w");
            if (fd == 0) begin $display("  ERROR: cannot write net/sim_%s.hex", lname[l]); $finish; end
            for (int i = 0; i < NPIX * l_cout[l]; i++) begin
                $fwrite(fd, "%04h\n", nxt[i]);
                fm[i] = nxt[i];
            end
            $fclose(fd);

            $display("  %-6s %3d -> %-3d shift %2d %-6s %0d pass(es)  %7d values  %0d mismatches  %0d negative  @ %0t",
                     lname[l], l_cin[l], l_cout[l], l_shift[l], l_lin[l] ? "linear" : "relu",
                     l_npass[l], NPIX * l_cout[l], layer_err, negatives, $time);
            errors += layer_err;
        end
    endtask

    initial begin
        logic [31:0] v;
        int fd, n, s;
        string nm;
        errors = 0; total_pass = 0;
        stall_in = 20; stall_out = 20;
        awvalid=0; wvalid=0; bready=0; arvalid=0; rready=0;
        awaddr='0; wdata='0; wstrb='0; araddr='0;
        wt_tvalid=0; wt_tdata='0;
        s_tvalid=0; s_tlast=0; s_tdata='0; m_tready=0;

        fd = $fopen("net/cfg.txt", "r");
        if (fd == 0) begin
            $display("FAIL  net/cfg.txt missing -- run python model\\golden_network.py first");
            $finish;
        end
        for (int l = 0; l < N_LAYERS; l++) begin
            n = $fscanf(fd, "%s %d %d %d %d %d\n", nm, l_cin[l], l_cout[l], l_shift[l],
                        l_lin[l], l_npass[l]);
            if (n != 6) begin $display("FAIL  net/cfg.txt line %0d unreadable", l + 1); $finish; end
            lname[l] = nm;
        end
        $fclose(fd);
        fd = $fopen("net/sat.txt", "r");
        n = 0;
        while ($fscanf(fd, "%d\n", s) == 1) begin sat_exp[n] = s; n++; end
        $fclose(fd);

        $readmemh("net/input.hex", fm);
        if ($isunknown(fm[NPIX * 6 - 1])) begin
            $display("FAIL  net/input.hex missing or short"); $finish;
        end

        rst_n = 0;
        repeat (4) @(posedge clk);
        rst_n = 1;
        @(negedge clk);

        $display("network golden: vfi_top %0dx%0d MAX_CH=%0d N_OUT=%0d ACT_MODE=1, trained weights, chained, %0d%% stalls",
                 IMG_W, IMG_H, MAX_CH, N_OUT, stall_in);
        axil_read(A_ID, v);
        if (v !== ID_MAGIC) begin $display("  ERROR: ID reads %08h", v); errors++; end

        for (int l = 0; l < N_LAYERS; l++) run_layer(l);

        axil_read(A_FRAMES, v);
        if (v !== total_pass) begin
            $display("  ERROR: FRAMES = %0d, expected %0d", v, total_pass); errors++;
        end
        $display("protocol: S_AXIS %0d beats / %0d pkts / %0d violations;  M_AXIS %0d / %0d / %0d",
                 in_beats, in_pkts, in_err, out_beats, out_pkts, out_err);
        errors += in_err + out_err;

        $display("");
        if (errors == 0)
            $display("PASS  all %0d layers bit-exact, %0d passes chained, FRAMES=%0d", N_LAYERS, total_pass, v);
        else
            $display("FAIL  %0d errors", errors);
        $finish;
    end

    initial begin
        #(64'd20_000_000_000);      // 20 s simulated; the run needs ~0.2 s
        $display("FAIL  timeout at out_idx=%0d", out_idx);
        $finish;
    end

endmodule
