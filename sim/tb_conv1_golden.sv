`timescale 1ns / 1ps
//
// conv1 on vfi_top with the TRAINED weights, checked value-for-value against the
// Python golden model (model/golden_conv1.py).
//
// Every other testbench here proves the arithmetic with random stimulus. This one
// runs the real layer the board will run: the exported conv1 kernels, the
// held-out input frame pair, the trained shift -- at the bitstream's exact
// configuration (112x112, MAX_CH=128, N_OUT=24), driven over AXI-Lite and
// AXI-Stream the way host/vfi_driver.py drives it, with stalls on both sides.
//
// conv1 has 32 output channels and the MAC array is 24 wide, so it runs as two
// passes back to back with no reset between them, exactly as run_layer() does:
//
//     pass 0: kernels 0..23
//     pass 1: kernels 24..31 + 16 zero kernels
//
// Generate the inputs first (they live in build/conv1/, and xsim runs in build/):
//
//     python model\golden_conv1.py
//     .\sim\run_xsim.ps1 -Tb tb_conv1_golden -Sources <rtl list, see conv1 notes>
//     python model\golden_conv1.py --check-sim
//
// The testbench compares in-line and also dumps every output beat to
// build/conv1/sim_pass<p>.hex, so the final comparison is made by Python
// against the golden model, independently of this file's own checker.
//
module tb_conv1_golden;

    localparam int DATA_W   = 16;
    localparam int WEIGHT_W = 16;
    localparam int IMG_W    = 112;
    localparam int IMG_H    = 112;
    localparam int MAX_CH   = 128;
    localparam int N_OUT    = 24;
    localparam int CH_IN    = 6;
    localparam int N_PASS   = 2;
    localparam int NPIX     = IMG_W * IMG_H;
    localparam int N_IN     = NPIX * CH_IN;
    localparam int N_WT     = N_OUT * CH_IN * 9;
    localparam int N_OUTB   = NPIX * N_OUT;
    // ACT_MODE stays 1, as the bitstream is built. TB_ACT=0 selects linear the
    // way the host does it for the flow layer -- CFG[14] over AXI-Lite -- and
    // exposes every negative value; generate the matching golden files with
    // `python model\golden_conv1.py --act linear`.
`ifdef TB_ACT
    localparam int ACT      = `TB_ACT;
`else
    localparam int ACT      = 1;
`endif
    localparam int LINEAR   = (ACT == 0);

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

    int in_err, in_beats, in_pkts;
    axis_checker #(.NAME ("S_AXIS"), .DATA_W (DATA_W), .PKT_LEN (N_IN)) u_chk_in (
        .clk (clk), .rst_n (rst_n),
        .tdata (s_tdata), .tlast (s_tlast), .tvalid (s_tvalid), .tready (s_tready),
        .errors (in_err), .beats (in_beats), .packets (in_pkts)
    );
    int out_err, out_beats, out_pkts;
    axis_checker #(.NAME ("M_AXIS"), .DATA_W (DATA_W), .PKT_LEN (N_OUTB)) u_chk_out (
        .clk (clk), .rst_n (rst_n),
        .tdata (m_tdata), .tlast (m_tlast), .tvalid (m_tvalid), .tready (m_tready),
        .errors (out_err), .beats (out_beats), .packets (out_pkts)
    );

    // Golden data, in beat order, from model/golden_conv1.py.
    logic [DATA_W-1:0]   img [N_IN];
    logic [WEIGHT_W-1:0] wts [N_PASS][N_WT];
    logic [DATA_W-1:0]   exp [N_PASS][N_OUTB];
    logic [7:0]          cfg [1 + N_PASS];      // shift, sat_sticky per pass

    int errors, checked, nonzero, negative, out_idx, stall_in, stall_out;

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
    task automatic dma_weights(int p);
        int i;
        bit pending;
        begin
            i = 0; pending = 0;
            while (i < N_WT) begin
                if (!pending) begin
                    if ($urandom_range(99) < stall_in) begin
                        wt_tvalid = 0; wt_tdata = 'x;
                        @(posedge clk); @(negedge clk); continue;
                    end
                    wt_tvalid = 1; wt_tdata = wts[p][i]; pending = 1;
                end
                @(posedge clk);
                if (wt_tready) begin i++; pending = 0; end
                @(negedge clk);
            end
            wt_tvalid = 0; wt_tdata = 'x;
            @(posedge clk); @(negedge clk);
        end
    endtask

    task automatic dma_frame(int p, int fd);
        int in_idx;
        bit pending;
        begin
            in_idx = 0; out_idx = 0;
            fork
                begin
                    pending = 0;
                    while (in_idx < N_IN) begin
                        if (!pending) begin
                            if ($urandom_range(99) < stall_in) begin
                                s_tvalid = 0; s_tdata = 'x; s_tlast = 0;
                                @(posedge clk); @(negedge clk); continue;
                            end
                            s_tvalid = 1;
                            s_tdata  = img[in_idx];
                            s_tlast  = (in_idx == N_IN - 1);
                            pending  = 1;
                        end
                        @(posedge clk);
                        if (s_tready) begin in_idx++; pending = 0; end
                        @(negedge clk);
                    end
                    s_tvalid = 0; s_tdata = 'x; s_tlast = 0;
                end
                begin
                    while (out_idx < N_OUTB) begin
                        m_tready = ($urandom_range(99) >= stall_out);
                        @(posedge clk);
                        if (m_tvalid && m_tready) begin
                            $fwrite(fd, "%04h\n", m_tdata);
                            if (m_tdata !== exp[p][out_idx]) begin
                                errors++;
                                if (errors <= 10)
                                    $display("  MISMATCH pass %0d (row %0d, col %0d, ch %0d): got %0d exp %0d",
                                             p, (out_idx / N_OUT) / IMG_W, (out_idx / N_OUT) % IMG_W,
                                             p * N_OUT + out_idx % N_OUT,
                                             $signed(m_tdata), $signed(exp[p][out_idx]));
                            end
                            if (m_tdata != 0) nonzero++;
                            if ($signed(m_tdata) < 0) negative++;
                            if (m_tlast != (out_idx == N_OUTB - 1)) begin
                                errors++;
                                $display("  ERROR: TLAST=%0b on beat %0d of %0d", m_tlast, out_idx, N_OUTB);
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

    task automatic run_pass(int p);
        logic [31:0] v;
        int fd, shift;
        string fn;
        begin
            shift = cfg[0];
            axil_write(A_CFG, 32'(LINEAR) << 14 | 32'(shift) << 8 | 32'(CH_IN));
            axil_read(A_CFG, v);
            if (v[7:0] !== 8'(CH_IN) || v[13:8] !== 6'(shift) || v[14] !== 1'(LINEAR)) begin
                $display("  ERROR: CFG readback %08h", v); errors++;
            end
            axil_write(A_CTRL, 32'h1);      // sat_clear

            dma_weights(p);
            axil_read(A_STATUS, v);
            if (!v[0]) begin
                $display("  ERROR: weights_loaded not set after %0d kernel beats", N_WT); errors++;
            end

            fn = $sformatf("conv1/sim_pass%0d.hex", p);
            fd = $fopen(fn, "w");
            if (fd == 0) begin $display("  ERROR: cannot open %s", fn); $finish; end
            dma_frame(p, fd);
            $fclose(fd);

            axil_read(A_STATUS, v);
            $display("  pass %0d: kernels %0d..%0d  shift=%0d  status=%02h  sat_sticky=%0b (golden %0d)  @ %0t",
                     p, p * N_OUT, p * N_OUT + N_OUT - 1, shift, v[7:0], v[1], cfg[1 + p], $time);
            if (v[1] !== cfg[1 + p][0]) begin
                $display("  ERROR: sat_sticky disagrees with the golden model"); errors++;
            end
        end
    endtask

    initial begin
        logic [31:0] v;
        errors = 0; checked = 0; nonzero = 0; negative = 0;
        stall_in = 25; stall_out = 25;
        awvalid=0; wvalid=0; bready=0; arvalid=0; rready=0;
        awaddr='0; wdata='0; wstrb='0; araddr='0;
        wt_tvalid=0; wt_tdata='0;
        s_tvalid=0; s_tlast=0; s_tdata='0; m_tready=0;

        $readmemh("conv1/input.hex",   img);
        $readmemh("conv1/w_pass0.hex", wts[0]);
        $readmemh("conv1/w_pass1.hex", wts[1]);
        $readmemh("conv1/exp_pass0.hex", exp[0]);
        $readmemh("conv1/exp_pass1.hex", exp[1]);
        $readmemh("conv1/cfg.hex",     cfg);
        if ($isunknown(img[N_IN-1]) || $isunknown(exp[1][N_OUTB-1]) || $isunknown(cfg[N_PASS])) begin
            $display("FAIL  golden files missing or short -- run python model\\golden_conv1.py first");
            $finish;
        end

        rst_n = 0;
        repeat (4) @(posedge clk);
        rst_n = 1;
        @(negedge clk);

        $display("conv1 golden: vfi_top %0dx%0d MAX_CH=%0d N_OUT=%0d %s, trained weights, %0d%% stalls",
                 IMG_W, IMG_H, MAX_CH, N_OUT, LINEAR ? "linear (CFG[14]=1)" : "ReLU", stall_in);
        axil_read(A_ID, v);
        if (v !== ID_MAGIC) begin
            $display("  ERROR: ID reads %08h", v); errors++;
        end

        for (int p = 0; p < N_PASS; p++) run_pass(p);

        axil_read(A_FRAMES, v);
        if (v !== N_PASS) begin
            $display("  ERROR: FRAMES = %0d, expected %0d", v, N_PASS); errors++;
        end
        $display("protocol: S_AXIS %0d beats / %0d pkts / %0d violations;  M_AXIS %0d / %0d / %0d",
                 in_beats, in_pkts, in_err, out_beats, out_pkts, out_err);
        errors += in_err + out_err;

        $display("");
        if (errors == 0)
            $display("PASS  %0d beats match the golden model (%0d nonzero, %0d negative), %0d passes",
                     checked, nonzero, negative, N_PASS);
        else
            $display("FAIL  %0d errors", errors);
        $finish;
    end

    initial begin
        #(64'd10 * N_PASS * (N_OUTB + N_IN) * 3 + 64'd1000000);
        $display("FAIL  timeout at out_idx=%0d", out_idx);
        $finish;
    end

endmodule
