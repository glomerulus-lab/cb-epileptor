import pickle
import os
import numpy as np
from brian2 import uS, start_scope
import params
import config
import data_processing
import synch as syn
import plotting.sync_heatmap as ps
from simulation.core import run_sim

DATA_DIR = config.DATA_DIR


def synchrony_sweep(quick=False, plot_mode='both'):
    """
    Run simulation across a 2D parameter grid and plot synchrony chi heatmaps

    plot_mode: 'chi' for mean only, 'sd' for SD only, 'both' for both
    """
    param1_values = np.linspace(0.05, 0.5, 8)
    param2_values = np.linspace(0.05, 0.5, 8)

    n_realizations = 1

    chi_grid = np.full((len(param2_values), len(param1_values), n_realizations), np.nan)

    total = len(param1_values) * len(param2_values) * n_realizations
    count = 0

    for j, p2 in enumerate(param2_values):
        for i, p1 in enumerate(param1_values):
            for k in range(n_realizations):
                count += 1
                print(f"[{count}/{total}] COUPLING_STRENGTH={p1:.3f}, G_INTER={p2:.3f}, realization {k+1}")

                params.COUPLING_STRENGTH = p1
                params.G_INTER = p2 * uS

                start_scope()
                run_sim()

                data = data_processing.load_sim_data()
                x1 = data['results']['x1']
                chi, _ = syn.autocorelate(x1)
                chi_grid[j, i, k] = chi
                print(f"  chi = {chi:.4f}")

    # Save raw results
    sweep_results = {
        'chi_grid': chi_grid,
        'param1_values': param1_values,
        'param2_values': param2_values,
        'param1_name': 'COUPLING_STRENGTH',
        'param2_name': 'G_INTER',
        'n_realizations': n_realizations,
    }
    sweep_path = os.path.join(DATA_DIR, 'sweep_results.pkl')
    with open(sweep_path, 'wb') as f:
        pickle.dump(sweep_results, f)
    print(f"Sweep data saved to {sweep_path}")

    chi_mean = np.nanmean(chi_grid, axis=2)
    chi_sd = np.nanstd(chi_grid, axis=2)
    p1_label = r'coupling strength'
    p2_label = r'$g_{inter}$ ($\mu$S)'

    if plot_mode == 'chi':
        ps.plot_synchrony_single(chi_mean, param1_values, param2_values,
                                 p1_label, p2_label,
                                 title=r'Synchrony $\chi$',
                                 vmin=0, vmax=1,
                                 save_name='synchrony_chi_mean.png')
    elif plot_mode == 'sd':
        sd_max = np.nanmax(chi_sd) if np.nanmax(chi_sd) > 0 else 0.15
        ps.plot_synchrony_single(chi_sd, param1_values, param2_values,
                                 p1_label, p2_label,
                                 title=r'SD of $\chi$',
                                 vmin=0, vmax=sd_max,
                                 save_name='synchrony_chi_sd.png')
    else:
        ps.plot_synchrony(chi_mean, chi_sd,
                          param1_values, param2_values,
                          p1_label, p2_label)
