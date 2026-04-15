from brian2 import *
import params

# Setup timed arrays
x_naught_vals = [-3.5]
coupling_vals = [0.2]

# x_naught_vals = [-4.5, -3.5, -3.5, -4.5, -4.5]
# coupling_vals = [0, 0, 0.2, 0.09, 0]


# G_inter_vals = [2] * uS
# G_intra_vals = [1] * uS

G_inter_vals = [1, 1, 4, 4, 1] * uS
G_intra_vals = [1, 4, 4, 1, 1] * uS

# G_inter_vals = [1, 1, 2, 2, 1] * uS
# G_intra_vals = [1, 2, 2, 1, 1] * uS

x_naught_dt = params.SIM_DURATION//len(x_naught_vals)
coupling_dt = params.SIM_DURATION//len(coupling_vals)

G_inter_dt = params.SIM_DURATION//len(G_inter_vals)
G_intra_dt = params.SIM_DURATION//len(G_intra_vals)

timed_x_naught = TimedArray(x_naught_vals, dt=x_naught_dt)
timed_coupling_strength = TimedArray(coupling_vals, dt=coupling_dt)

timed_G_inter = TimedArray(G_inter_vals, dt=G_inter_dt)
timed_G_intra = TimedArray(G_intra_vals, dt=G_intra_dt)
