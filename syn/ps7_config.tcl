# PS7 configuration for the PYNQ-Z2.
#
# Only the settings that affect the PL are set here, deliberately.
#
# An earlier version replayed all 525 PCW_* properties extracted from the
# project's own VFI2.bd. That fails for two reasons: most of them are derived,
# read-only outputs (PCW_ACT_*, computed FREQMHZ values, the MIO tree) which
# Vivado refuses to set, and some MIO values contain characters that break Tcl
# parsing when replayed as a dict -- "invalid command name '\''GPIO#Quad".
#
# It is also unnecessary. For a PYNQ overlay the DDR and MIO configuration is
# applied by the FSBL and u-boot at boot time, not by the bitstream: the board
# is already running Linux when an overlay is loaded. What the overlay actually
# needs from PS7 is the PL clock, the GP master used for control, and the HP
# slave the DMAs use to reach DDR.
#
# The DDR part and bank voltages are set anyway, matching the values the working
# VFI2.bd used (MT41J256M16 RE-125, 3.3 V / 1.8 V, 50 MHz crystal), so the block
# design is self-describing and correct if it is ever built for a standalone
# flow that does run ps7_init.

# PL clock is 75 MHz, not 100. At 100 MHz the design misses setup by 2.793 ns
# post-route, and the failing path is 82% routing with only 4 logic levels:
#
#   u_lb/chq_reg -> g_mac[16]/prod_q_reg   8.672ns = 1.551 logic + 7.121 route
#
# That is physical, not logical -- the 144-bit window is broadcast to 24 MACs
# and 216 DSPs span the die, so more pipeline stages inside the MAC cannot help.
# Fixing it properly means pipelining the broadcast itself (two register stages
# between line_buffer_mc and the MAC array, splitting the route). Worth doing
# when frame rate actually matters; 75 MHz costs ~25% throughput (conv3 43 -> 57
# ms) and is not the binding constraint while there is no trained model.
#
# Post-synthesis slack said +1.126 ns for the same design. Estimated routing is
# optimistic; only post-route WNS counts.
proc vfi_configure_ps7 {cell} {
    set_property -dict [list \
        CONFIG.PCW_EN_CLK0_PORT             {1} \
        CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {75} \
        CONFIG.PCW_USE_M_AXI_GP0            {1} \
        CONFIG.PCW_USE_S_AXI_HP0            {1} \
        CONFIG.PCW_S_AXI_HP0_DATA_WIDTH     {64} \
        CONFIG.PCW_CRYSTAL_PERIPHERAL_FREQMHZ {50} \
        CONFIG.PCW_UIPARAM_DDR_PARTNO       {MT41J256M16 RE-125} \
        CONFIG.PCW_UIPARAM_DDR_MEMORY_TYPE  {DDR 3} \
        CONFIG.PCW_UIPARAM_DDR_BUS_WIDTH    {16 Bit} \
        CONFIG.PCW_PRESET_BANK0_VOLTAGE     {LVCMOS 3.3V} \
        CONFIG.PCW_PRESET_BANK1_VOLTAGE     {LVCMOS 1.8V} \
    ] $cell
}
