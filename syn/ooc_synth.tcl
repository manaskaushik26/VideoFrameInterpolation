# Out-of-context synthesis of a single module, for resource numbers.
#
#   vivado -mode batch -source syn/ooc_synth.tcl -tclargs <top> <rtl_dir> "P=V P=V ..."
#
# Reports utilisation only -- no placement, no bitstream. The point is to turn
# the DSP/BRAM/LUT budget into a measurement instead of a spreadsheet estimate,
# early enough that the numbers can still change the architecture.
#
# OOC mode is deliberate: it skips I/O buffer insertion, so a module with more
# ports than the device has pins still synthesises and still reports honest
# logic usage.

set top      [lindex $argv 0]
set rtl_dir  [lindex $argv 1]
set generics [expr {$argc > 2 ? [lindex $argv 2] : ""}]

set part xc7z020clg400-1

create_project -in_memory -part $part

set srcs [glob -nocomplain [file join $rtl_dir *.sv] [file join $rtl_dir * *.sv]]
if {[llength $srcs] == 0} { puts "ERROR: no sources under $rtl_dir"; exit 1 }
read_verilog -sv $srcs

set gargs {}
foreach g $generics { if {[string length $g]} { lappend gargs -generic $g } }

puts "=== synth $top ($part) generics: $generics ==="
if {[catch { synth_design -top $top -mode out_of_context -part $part {*}$gargs } err]} {
    puts "SYNTH FAILED: $err"
    exit 1
}

puts "=== UTILISATION: $top $generics ==="
report_utilization -hierarchical -hierarchical_depth 2

# Compact summary that survives grepping.
set luts  [get_property SLICE_LUTS_USED  [get_cells -hier -filter {IS_PRIMITIVE}] ]
set rpt [report_utilization -return_string]
foreach line [split $rpt "\n"] {
    if {[regexp {\|\s+(Slice LUTs|LUT as Logic|LUT as Memory|Slice Registers|Block RAM Tile|DSPs)\s+\|\s+(\d+)\s+\|} $line -> name used]} {
        puts [format "SUMMARY %-18s %s" $name $used]
    }
}

# Timing estimate at the target clock.
create_clock -period 10.000 -name clk [get_ports clk]
puts "=== TIMING (100 MHz target) ==="
report_timing_summary -delay_type max -max_paths 1 -no_header
exit 0
