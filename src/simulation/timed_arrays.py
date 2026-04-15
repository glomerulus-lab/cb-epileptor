from brian2 import *
import params

timed_x_naught = TimedArray(params.X_NAUGHT_VALS, dt=params.SIM_DURATION//len(params.X_NAUGHT_VALS))
timed_coupling_strength = TimedArray(params.COUPLING_VALS, dt=params.SIM_DURATION//len(params.COUPLING_VALS))
timed_G_inter = TimedArray(params.G_INTER_VALS, dt=params.SIM_DURATION//len(params.G_INTER_VALS))
timed_G_intra = TimedArray(params.G_INTRA_VALS, dt=params.SIM_DURATION//len(params.G_INTRA_VALS))
