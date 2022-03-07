from pymol import cmd

cmd.delete('all')
cmd.fetch('1GPWC 1GPWD')
cmd.color('grey80', '1GPW*')
cmd.remove('!(polymer)')
cmd.run('draw_network_pymol.py')
##MUTATE IN D#
cmd.wizard("mutagenesis")
cmd.get_wizard().do_select("resi 11 and chain C")
cmd.get_wizard().set_mode("ASP")
cmd.get_wizard().apply()

cmd.hide('lines', '*')

cmd.set_view((
    -0.091599651,    0.961403608,    0.259411007,\
    -0.907361209,    0.026737407,   -0.419488847,\
    -0.410239011,   -0.273805201,    0.869903326,\
     0.000195857,    0.000733256, -208.243789673,\
   -28.106067657,   12.463504791,   74.449974060,\
  -14447.857421875, 14864.273437500,   20.000000000))

draw_from_df('../data/dpcn.dfp', w2='prfar', w1='apo', color_sign=1)