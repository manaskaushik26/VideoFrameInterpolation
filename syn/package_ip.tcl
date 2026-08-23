# Package rtl/ as a Vivado IP into ip_repo/.
#
#   vivado -mode batch -source syn/package_ip.tcl
#   vivado -mode batch -source syn/package_ip.tcl -tclargs <ip_repo_dir>
#
# Why package rather than use `create_bd_cell -type module`:
#
#   - a packaged IP declares its AXI interfaces explicitly in IP-XACT, instead of
#     relying on Vivado inferring them from port names. Inference is silent when
#     it fails: the cell simply appears with loose pins that will not connect,
#     and nothing says why.
#   - packaging accepts SystemVerilog as the top, so rtl/axis/vfi_top_v.v -- a
#     wrapper that exists only because module references reject .sv tops -- is no
#     longer needed for the BD flow.
#   - the IP is versioned and usable from the GUI, so the block design can be
#     opened and inspected by hand.
#
# The build works either way; build.tcl prefers the packaged IP when ip_repo/
# exists and falls back to the module reference otherwise.

# ip_repo_paths expects a directory CONTAINING IP directories, not an IP
# directory itself. Packaging component.xml at the root of ip_repo/ leaves the
# core invisible to the catalog -- update_ip_catalog reports nothing and
# get_ipdefs finds nothing, with no error raised anywhere. So the core goes in
# ip_repo/vfi_top/ and ip_repo/ is the repository.
set ip_root [expr {$argc > 0 ? [lindex $argv 0] : \
                   [file join [file dirname [info script]] .. ip_repo]}]
set ip_root [file normalize $ip_root]
set ip_dir  [file join $ip_root vfi_top]
set repo   [file normalize [file join [file dirname [info script]] ..]]
set part   xc7z020clg400-1
set tmp    [file join $::env(LOCALAPPDATA) vfi_ip_tmp]

puts "=== packaging vfi_top -> $ip_dir (repo root $ip_root) ==="

file delete -force $tmp
file delete -force $ip_root
file mkdir $ip_dir

create_project -force vfi_ip $tmp -part $part

set rtl [glob -nocomplain [file join $repo rtl * *.sv] [file join $repo rtl *.sv]]
if {[llength $rtl] == 0} { error "no RTL under $repo/rtl" }
add_files -norecurse $rtl
set_property file_type SystemVerilog [get_files *.sv]
set_property top vfi_top [current_fileset]
update_compile_order -fileset sources_1

ipx::package_project -root_dir $ip_dir -vendor vfi -library user \
    -taxonomy /UserIP -import_files -set_current true

set core [ipx::current_core]
set_property name         vfi_top   $core
set_property display_name {VFI conv layer} $core
set_property description  {Runtime-configurable 3x3 convolution layer with AXI-Lite control} $core
set_property version      1.0       $core
set_property core_revision 1        $core

# Associate every stream and the control interface with the clock, and the clock
# with the reset. Vivado usually infers this from aclk/aresetn, but stating it
# is the point of packaging -- an inferred association that silently fails leaves
# the interfaces unclocked and the BD refuses to connect them.
foreach busif {s_axi s_axis m_axis s_wt} {
    if {[llength [ipx::get_bus_interfaces $busif -of_objects $core]] > 0} {
        ipx::associate_bus_interfaces -busif $busif -clock aclk $core
    } else {
        puts "WARNING: bus interface '$busif' was not inferred during packaging"
    }
}

set rstif [ipx::get_bus_interfaces aresetn -of_objects $core]
if {[llength $rstif] > 0} {
    set p [ipx::get_bus_parameters POLARITY -of_objects $rstif]
    if {[llength $p] > 0} { set_property value ACTIVE_LOW $p }
}

ipx::create_xgui_files $core
ipx::update_checksums  $core
ipx::check_integrity   $core
ipx::save_core         $core

puts "=== packaged interfaces ==="
foreach b [ipx::get_bus_interfaces -of_objects $core] {
    puts "IP_BUSIF [get_property name $b]"
}

close_project
file delete -force $tmp
puts "=== PACKAGE OK: core $ip_dir, repo root $ip_root ==="
exit 0
