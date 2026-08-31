# Out-of-context synthesis + worst-path timing detail for one module.
#
#   vivado -mode batch -source syn/ooc_timing.tcl -tclargs <top> <rtl_dir> "P=V ..." <period_ns>
#
# Post-synthesis timing is estimated, not routed, so treat the numbers as
# indicative. A violation of several nanoseconds against the period is real; a
# few hundred picoseconds is noise.
#
# A POSITIVE NUMBER HERE IS NOT TIMING CLOSURE. This design measured +1.126 ns
# post-synthesis at 100 MHz and -2.793 ns post-route, because the critical path
# is 82% routing -- exactly what an out-of-context estimate cannot model. Only
# the post-route WNS from syn/build.tcl decides whether a clock is met.
#
# Default period is 13.333 ns (75 MHz), the clock the design actually ships at.

# Period before generics: PowerShell silently drops an empty-string argument,
# which shifts every later -tclargs position by one.
#
# Also note -generic corrupts any parameter used as a part-select width
# (`x[i*W +: W]`) -- Vivado passes it in a form that is no longer a usable
# constant, and synthesis fails with "range width must be a positive integer".
# Override only parameters used in plain arithmetic, such as IMG_W or N_OUT.
set top      [lindex $argv 0]
set rtl_dir  [lindex $argv 1]
set period   [expr {$argc > 2 ? [lindex $argv 2] : 13.333}]
set generics [expr {$argc > 3 ? [lindex $argv 3] : ""}]

create_project -in_memory -part xc7z020clg400-1
set srcs [glob -nocomplain [file join $rtl_dir *.sv] [file join $rtl_dir * *.sv]]
read_verilog -sv $srcs

set gargs {}
foreach g $generics { if {[string length $g]} { lappend gargs -generic $g } }

synth_design -top $top -mode out_of_context -part xc7z020clg400-1 {*}$gargs
create_clock -period $period -name clk [get_ports clk]

puts "=== WORST PATHS ($top, period ${period}ns) ==="
report_timing -delay_type max -max_paths 3 -nworst 3 -path_type full -input_pins
exit 0
