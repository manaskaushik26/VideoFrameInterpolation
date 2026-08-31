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

# Default build location is OUTSIDE the repo, and that is deliberate.
#
# This repository lives under OneDrive. A Vivado project is tens of thousands of
# small files, and the sync client holds handles on them while Vivado is writing:
# the build fails partway with "[Common 17-1293] The path '.../vfi.cache/wt'
# already exists, is a directory, but is not writable", and xelab hits the same
# thing on its scratch directories. Building to LOCALAPPDATA avoids it entirely.
#
# Pass an explicit path as the first argument to override.
set default_out [file join $::env(LOCALAPPDATA) vfi_build]
set out_dir [expr {$argc > 0 ? [lindex $argv 0] : $default_out}]
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
set rtl [glob -nocomplain [file join $repo rtl * *.sv] [file join $repo rtl *.sv]                           [file join $repo rtl * *.v]  [file join $repo rtl *.v]]
if {[llength $rtl] == 0} { error "no RTL found under $repo/rtl" }
add_files -norecurse $rtl
set_property file_type SystemVerilog [get_files *.sv]

# Width mismatches (Synth 8-689) and port-count mismatches (8-7023) are the two
# warning classes that between them let the superseded design synthesise a
# netlist computing a constant zero and still report success. They must fail the
# build.
#
# But they cannot simply be promoted with set_msg_config: that is project-wide,
# and Xilinx's own generated IP trips 8-7023 routinely and legitimately -- the
# processing_system7 wrapper declares 685 connections and gives 672, and
# smartconnect does the same. Promoting it globally fails Xilinx IP, not our
# RTL. (Our RTL synthesised clean in that run; only the IP runs failed.)
#
# So the check is applied after synthesis, scoped to files under rtl/. See
# check_our_warnings below.

# ---------------------------------------------------------------- block design
create_bd_design $bd_name

set ps [create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7]
source [file join $repo syn ps7_config.tcl]
# One 75 MHz FCLK, one HP slave for the DMAs to reach DDR, one GP master for
# control. See syn/ps7_config.tcl for why only PL-facing settings are set, and
# for why the clock is 75 MHz and not 100. (The reset cell below is still named
# rst_100 -- a stale name, not a stale setting; renaming a working BD cell was
# not worth the churn.)
vfi_configure_ps7 $ps

set rst [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_100]

# The IP under test.
#
# Preferred: the packaged IP from ip_repo/, built by syn/package_ip.tcl. It
# declares its AXI interfaces explicitly in IP-XACT rather than relying on
# Vivado inferring them from port names, and it takes the SystemVerilog top
# directly.
#
# Fallback: a module reference through vfi_top_v.v, a plain Verilog wrapper that
# exists because module references reject SystemVerilog tops
# ("[filemgmt 56-195] ... not allowed as the top file"). Kept so the build works
# from a clean checkout without running package_ip.tcl first.
set ip_repo [file join $repo ip_repo]
set use_packaged 0
if {[file isdirectory $ip_repo]} {
    # [list $ip_repo], not $ip_repo. ip_repo_paths is a Tcl list property, and
    # this repository's path contains a space ("VFI FPGA"), so passing the bare
    # string splits it into two nonexistent entries:
    #   {.../VFI} {.../VFI FPGA/FPGA/ip_repo}
    # The catalog then finds nothing and reports no error at all.
    set_property ip_repo_paths [list $ip_repo] [current_project]
    update_ip_catalog -rebuild
    # Match on VLNV, not NAME: an ipdef's NAME is not the module name.
    if {[llength [get_ipdefs -filter {VLNV =~ vfi:user:vfi_top:*}]] > 0} { set use_packaged 1 }
}

if {$use_packaged} {
    puts "=== using packaged IP from $ip_repo ==="
    set dut [create_bd_cell -type ip -vlnv vfi:user:vfi_top:1.0 vfi]
} else {
    puts "=== ip_repo not found; falling back to module reference ==="
    set dut [create_bd_cell -type module -reference vfi_top_v vfi]
}
set_property -dict [list \
    CONFIG.N_OUT  $n_out \
    CONFIG.MAX_CH $max_ch \
] $dut

# Feature-map DMA: MM2S feeds the layer, S2MM collects the result.
# Buffer Length Register width. THE DEFAULT (14) IS UNUSABLE HERE: it caps a
# single transfer at 16,383 bytes, while every feature map in this design is far
# larger -- 112*112*6*2 = 150,528 bytes on the way in, 112*112*24*2 = 602,112 on
# the way out, and 112*112*128*2 = 3,211,264 for conv4's input. Only the smallest
# weight set (conv1, 2,592 bytes) fits, so bring-up would pass the ID, CFG and
# first kernel load and then fail on the first feature map.
#
# 26 bits is the IP maximum and gives a 64 MB ceiling; 23 would cover the 3.2 MB
# worst case. The cost is a few registers in the DMA. host/vfi_driver.py issues
# one transfer per buffer with no chunking and should stay that way -- splitting
# a feature map in software would put a seam mid-frame that the line buffer's
# counters know nothing about.
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
    CONFIG.c_sg_length_width {26} \
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
    CONFIG.c_sg_length_width {26} \
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

# Interconnects, built explicitly rather than by apply_bd_automation.
#
# The automation route failed here with "[Ip 78-87] Error found in procedure
# get_rule_options" once more than one master had been connected, even though
# S_AXI_HP0 was enabled and present on PS7 -- verified by probing the configured
# cell directly. Building the two smartconnects by hand is deterministic, and it
# puts the topology in the script instead of leaving it implicit in a tool rule.
#
#   control: ps7/M_AXI_GP0 -> { vfi/s_axi, dma_fm lite, dma_wt lite }
#   memory:  { dma_fm MM2S, dma_fm S2MM, dma_wt MM2S } -> ps7/S_AXI_HP0

# SmartConnect ships a SystemC/TLM simulation model, and generating it is a
# failure path this build does not need. A run died here with
#
#   [IP_Flow 19-1747] Failed to deliver file '.../xlconstant_v1_1.ttcl'
#   [IP_Flow 19-3505] Failed to generate 'SystemC Simulation' outputs
#   [Vivado 12-4756] Launch of runs aborted
#
# on an internal xlconstant inside smc_mem's clock-map, while the file itself was
# present and readable both before and after. Nothing about a bitstream needs a
# SystemC model, so select the RTL simulation model and the whole path goes away.
#
# Wrapped in catch: if a future IP version drops the property, that must not be
# what breaks the build.
proc vfi_use_rtl_sim_model {cell} {
    if {[catch {set_property SELECTED_SIM_MODEL rtl $cell} err]} {
        puts "note: SELECTED_SIM_MODEL not applied ($err)"
    }
}

set smc_ctrl [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smc_ctrl]
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI {3}] $smc_ctrl
vfi_use_rtl_sim_model $smc_ctrl
connect_bd_net [get_bd_pins ps7/FCLK_CLK0]              [get_bd_pins smc_ctrl/aclk]
connect_bd_net [get_bd_pins rst_100/peripheral_aresetn] [get_bd_pins smc_ctrl/aresetn]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0]              [get_bd_pins ps7/M_AXI_GP0_ACLK]
connect_bd_intf_net [get_bd_intf_pins ps7/M_AXI_GP0]    [get_bd_intf_pins smc_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins smc_ctrl/M00_AXI] [get_bd_intf_pins vfi/s_axi]
connect_bd_intf_net [get_bd_intf_pins smc_ctrl/M01_AXI] [get_bd_intf_pins dma_fm/S_AXI_LITE]
connect_bd_intf_net [get_bd_intf_pins smc_ctrl/M02_AXI] [get_bd_intf_pins dma_wt/S_AXI_LITE]

set smc_mem [create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect:1.0 smc_mem]
set_property -dict [list CONFIG.NUM_SI {3} CONFIG.NUM_MI {1}] $smc_mem
vfi_use_rtl_sim_model $smc_mem
connect_bd_net [get_bd_pins ps7/FCLK_CLK0]              [get_bd_pins smc_mem/aclk]
connect_bd_net [get_bd_pins rst_100/peripheral_aresetn] [get_bd_pins smc_mem/aresetn]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0]              [get_bd_pins ps7/S_AXI_HP0_ACLK]
connect_bd_intf_net [get_bd_intf_pins dma_fm/M_AXI_MM2S] [get_bd_intf_pins smc_mem/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins dma_fm/M_AXI_S2MM] [get_bd_intf_pins smc_mem/S01_AXI]
connect_bd_intf_net [get_bd_intf_pins dma_wt/M_AXI_MM2S] [get_bd_intf_pins smc_mem/S02_AXI]
connect_bd_intf_net [get_bd_intf_pins smc_mem/M00_AXI]   [get_bd_intf_pins ps7/S_AXI_HP0]


assign_bd_address
regenerate_bd_layout
validate_bd_design
save_bd_design

# ---------------------------------------------------------------- IP generation
# Generate the block design's IP serially and up front, with retries.
#
# Left to launch_runs, this happens implicitly while the run is preparing
# sub-designs, in parallel, and on this machine it fails intermittently:
#
#   [IP_Flow 19-1747] Failed to deliver file '<vivado install>/data/ip/...'
#   [IP_Flow 19-167]  Failed to deliver one or more file(s).
#   [Vivado 12-4756]  Launch of runs aborted
#
# A DIFFERENT file fails each run -- xlconstant_v1_1.ttcl, then ps7's xdc.xit and
# proc_sys_reset's board.xit -- and every one of them is present, non-empty and
# readable immediately before and after. Ruled out: disk space (522 GB free),
# cloud-tiered files (plain Archive attributes, no reparse point), mark-of-the-web
# (no Zone.Identifier stream), leftover Vivado processes, and Controlled Folder
# Access. What is left is transient contention on the install directory, with
# Defender real-time scanning the likeliest culprit.
#
# So: do it once, serially, and retry rather than aborting the whole build. Only
# synthesis and implementation targets -- nothing here needs a simulation model,
# which is also why the SmartConnects select the RTL sim model above.
proc vfi_generate_ip {bd_file {tries 3}} {
    for {set i 1} {$i <= $tries} {incr i} {
        if {![catch {generate_target {synthesis implementation} $bd_file} err]} {
            puts "=== IP generation OK (attempt $i) ==="
            return
        }
        puts "WARNING: IP generation attempt $i of $tries failed: $err"
        after 2000
    }
    puts "ERROR: IP generation failed after $tries attempts"
    exit 1
}
vfi_generate_ip [get_files ${bd_name}.bd]

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
# Fail on width/port-count warnings, but only for our own sources. Xilinx IP
# legitimately triggers 8-7023, so a project-wide promotion is not usable.
proc check_our_warnings {logfile rtl_dir} {
    if {![file exists $logfile]} { return 0 }
    set fh [open $logfile r]
    set txt [read $fh]
    close $fh
    set bad 0
    foreach line [split $txt "
"] {
        if {[regexp {Synth 8-(689|7023)} $line] && [string match -nocase "*${rtl_dir}*" $line]} {
            puts "OURS-STRICT $line"
            incr bad
        }
    }
    return $bad
}

launch_runs synth_1 -jobs 4
wait_on_run synth_1
if {[get_property PROGRESS [get_runs synth_1]] != "100%"} {
    puts "SYNTHESIS FAILED"
    exit 1
}

set synth_log [file join $out_dir ${proj}.runs synth_1 runme.log]
set bad [check_our_warnings $synth_log [file join $repo rtl]]
if {$bad > 0} {
    puts "SYNTHESIS FAILED: $bad width/port-count warning(s) in our RTL"
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

# A bitstream is written even when timing fails, so the artifact existing proves
# nothing. Post-synthesis slack is an estimate and was optimistic here by nearly
# 4 ns; only post-route WNS/WHS count. Report the failure loudly rather than
# printing BUILD OK over it -- a build that says OK while the design cannot meet
# its clock is the same false success that produced the constant-zero bitstream.
set timing_ok 1
if {$wns < 0} {
    puts "SUMMARY TIMING FAILED setup WNS ${wns}ns"
    set timing_ok 0
}
if {$whs < 0} {
    puts "SUMMARY TIMING FAILED hold WHS ${whs}ns"
    set timing_ok 0
}

puts "=== WORST SETUP PATH ==="
report_timing -delay_type max -max_paths 1 -nworst 1 -path_type full

set bit [file join $out_dir ${proj}.runs impl_1 ${bd_name}_wrapper.bit]
if {[file exists $bit]} {
    file copy -force $bit [file join $out_dir vfi.bit]
    puts "SUMMARY BITSTREAM [file join $out_dir vfi.bit]"
} else {
    puts "SUMMARY BITSTREAM missing"
}

write_hw_platform -fixed -include_bit -force [file join $out_dir vfi.xsa]
puts "SUMMARY XSA [file join $out_dir vfi.xsa]"

if {!$timing_ok} {
    puts "=== BUILD FAILED: timing not met ==="
    exit 1
}
puts "=== BUILD OK ==="
exit 0
