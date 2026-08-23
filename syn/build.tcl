# Full build: RTL -> block design -> bitstream -> XSA, from nothing.
#
#   vivado -mode batch -source syn/build.tcl
#   vivado -mode batch -source syn/build.tcl -tclargs <out_dir> <n_out> <max_ch>
#
# Everything is generated. The project directory is disposable and gitignored --
# the sources of truth are rtl/, this script, and syn/ps7_config.tcl. That is
# deliberate: the superseded project's state lived in a .xpr nobody could
# reproduce, its top-level synthesis had never completed, and its checked-in
# bitstream was stale and built from a different block design.
#
# What this builds:
#
#     PS7 ---- AXI interconnect ---- vfi_top s_axi   (control)
#      |
#      +--- axi_dma_wt   MM2S ------ vfi_top s_wt    (kernels)
#      +--- axi_dma_fm   MM2S ------ vfi_top s_axis  (feature map in)
#                        S2MM ------ vfi_top m_axis  (feature map out)
#
# Two DMAs because three independent streams are needed and one AXI-DMA offers
# at most one MM2S and one S2MM. Both are in simple mode (no scatter-gather),
# matching what the PYNQ `pynq.lib.DMA` driver expects.

set out_dir [expr {$argc > 0 ? [lindex $argv 0] : "build/vivado"}]
set n_out   [expr {$argc > 1 ? [lindex $argv 1] : 24}]
set max_ch  [expr {$argc > 2 ? [lindex $argv 2] : 128}]

set part    xc7z020clg400-1
set repo    [file normalize [file join [file dirname [info script]] ..]]
set proj    vfi
set bd_name vfi_bd

puts "=== build: N_OUT=$n_out MAX_CH=$max_ch -> $out_dir ==="

file mkdir $out_dir
create_project -force $proj $out_dir -part $part

# ---------------------------------------------------------------- sources
set rtl [glob -nocomplain [file join $repo rtl * *.sv] [file join $repo rtl *.sv]]
if {[llength $rtl] == 0} { error "no RTL found under $repo/rtl" }
add_files -norecurse $rtl
set_property file_type SystemVerilog [get_files *.sv]

# Promote the two message classes that let this project build nothing. Width
# mismatches and port-count mismatches were only warnings in the superseded
# design, and between them they produced a netlist that computed a constant zero
# and still reported success.
set_msg_config -id {Synth 8-689}  -new_severity ERROR
set_msg_config -id {Synth 8-7023} -new_severity ERROR

# ---------------------------------------------------------------- block design
create_bd_design $bd_name

set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7]
source [file join $repo syn ps7_config.tcl]
vfi_configure_ps7 $ps
# One 100 MHz FCLK, one HP port for the DMAs to reach DDR, one GP master for
# control, and the GP slave the DMAs use.
set_property -dict [list \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_M_AXI_GP0_ENABLE_STATIC_REMAP {0} \
    CONFIG.PCW_EN_CLK0_PORT {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
] $ps

set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_100]

# The IP under test, as a module reference. Its ports are named so Vivado infers
# aclk/aresetn and the AXI interfaces automatically; see vfi_top's header.
set dut [create_bd_cell -type module -reference vfi_top vfi]
set_property -dict [list \
    CONFIG.N_OUT  $n_out \
    CONFIG.MAX_CH $max_ch \
] $dut

# Feature-map DMA: MM2S feeds the layer, S2MM collects the result.
set dma_fm [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 dma_fm]
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_sg_include_stscntrl_strm {0} \
    CONFIG.c_include_mm2s {1} \
    CONFIG.c_include_s2mm {1} \
    CONFIG.c_m_axi_mm2s_data_width {32} \
    CONFIG.c_m_axis_mm2s_tdata_width {16} \
    CONFIG.c_m_axi_s2mm_data_width {32} \
    CONFIG.c_s_axis_s2mm_tdata_width {16} \
    CONFIG.c_mm2s_burst_size {16} \
    CONFIG.c_s2mm_burst_size {16} \
] $dma_fm

# Weight DMA: MM2S only.
set dma_wt [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_dma:7.1 dma_wt]
set_property -dict [list \
    CONFIG.c_include_sg {0} \
    CONFIG.c_include_mm2s {1} \
    CONFIG.c_include_s2mm {0} \
    CONFIG.c_m_axi_mm2s_data_width {32} \
    CONFIG.c_m_axis_mm2s_tdata_width {16} \
    CONFIG.c_mm2s_burst_size {16} \
] $dma_wt

# ---------------------------------------------------------------- connections
# Streams, by hand: these are the connections the design is actually about, and
# leaving them to automation is how the old block design ended up with an 8-bit
# path where the model needed int16.
connect_bd_intf_net [get_bd_intf_pins dma_fm/M_AXIS_MM2S] [get_bd_intf_pins vfi/s_axis]
connect_bd_intf_net [get_bd_intf_pins vfi/m_axis]         [get_bd_intf_pins dma_fm/S_AXIS_S2MM]
connect_bd_intf_net [get_bd_intf_pins dma_wt/M_AXIS_MM2S] [get_bd_intf_pins vfi/s_wt]

# Clock and reset for everything.
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] \
               [get_bd_pins rst_100/slowest_sync_clk] \
               [get_bd_pins vfi/aclk] \
               [get_bd_pins dma_fm/s_axi_lite_aclk] [get_bd_pins dma_fm/m_axi_mm2s_aclk] \
               [get_bd_pins dma_fm/m_axi_s2mm_aclk] \
               [get_bd_pins dma_wt/s_axi_lite_aclk] [get_bd_pins dma_wt/m_axi_mm2s_aclk]
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins rst_100/ext_reset_in]
connect_bd_net [get_bd_pins rst_100/peripheral_aresetn] \
               [get_bd_pins vfi/aresetn] \
               [get_bd_pins dma_fm/axi_resetn] [get_bd_pins dma_wt/axi_resetn]

# Control path (PS -> AXI-Lite slaves) and memory path (DMAs -> DDR).
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config \
    [list Master {/ps7/M_AXI_GP0} Clk {/ps7/FCLK_CLK0}] [get_bd_intf_pins vfi/s_axi]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config \
    [list Master {/ps7/M_AXI_GP0} Clk {/ps7/FCLK_CLK0}] [get_bd_intf_pins dma_fm/S_AXI_LITE]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config \
    [list Master {/ps7/M_AXI_GP0} Clk {/ps7/FCLK_CLK0}] [get_bd_intf_pins dma_wt/S_AXI_LITE]

apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config \
    [list Slave {/ps7/S_AXI_HP0} Clk {/ps7/FCLK_CLK0}] [get_bd_intf_pins dma_fm/M_AXI_MM2S]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config \
    [list Slave {/ps7/S_AXI_HP0} Clk {/ps7/FCLK_CLK0}] [get_bd_intf_pins dma_fm/M_AXI_S2MM]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config \
    [list Slave {/ps7/S_AXI_HP0} Clk {/ps7/FCLK_CLK0}] [get_bd_intf_pins dma_wt/M_AXI_MM2S]

assign_bd_address
regenerate_bd_layout
validate_bd_design
save_bd_design

# ---------------------------------------------------------------- wrapper
make_wrapper -files [get_files ${bd_name}.bd] -top
add_files -norecurse [file join $out_dir ${proj}.gen sources_1 bd $bd_name hdl ${bd_name}_wrapper.v]
set_property top ${bd_name}_wrapper [current_fileset]
update_compile_order -fileset sources_1

# No user XDC: every physical pin belongs to PS7's fixed DDR/MIO interface, which
# needs no constraints, and the clock comes from FCLK_CLK0 and is constrained by
# the block design. If a port ever does reach a package pin, constrain it --
# do NOT reintroduce the old project's allow_unconstrained_io.tcl, which
# downgraded NSTD-1/UCIO-1 from error to warning and let unconstrained I/O
# through silently.

# ---------------------------------------------------------------- build
launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "SYNTHESIS FAILED"
    exit 1
}

launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "IMPLEMENTATION FAILED"
    exit 1
}

open_run impl_1
puts "=== POST-ROUTE ==="
report_utilization
report_timing_summary -delay_type min_max -max_paths 1 -no_header

set wns [get_property STATS.WNS [get_runs impl_1]]
set whs [get_property STATS.WHS [get_runs impl_1]]
puts "SUMMARY WNS $wns"
puts "SUMMARY WHS $whs"

set bit [file join $out_dir ${proj}.runs impl_1 ${bd_name}_wrapper.bit]
if {[file exists $bit]} {
    file copy -force $bit [file join $out_dir vfi.bit]
    puts "SUMMARY BITSTREAM [file join $out_dir vfi.bit]"
} else {
    puts "SUMMARY BITSTREAM missing"
}

write_hw_platform -fixed -include_bit -force [file join $out_dir vfi.xsa]
puts "SUMMARY XSA [file join $out_dir vfi.xsa]"
puts "=== BUILD OK ==="
exit 0
