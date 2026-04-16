import os
import numpy as np
import config
import params 
import datetime 
import pickle
from brian2 import *
from brian2tools import *

DATA_DIR = config.DATA_DIR
FIGURES_DIR = config.FIGURES_DIR
OUTPUT_DATA_FILE = config.OUTPUT_DATA_FILE

def create_spike_matrix_histo(spike_data, num_cells):
    spike_times = spike_data['t'] 
    neuron_indices = spike_data['i'] 

    duration = params.SIM_DURATION/second+params.TRANSIENT
    dt = 0.02  # 20ms per bin

    time_bins = np.arange(0, duration + dt, dt)
    neuron_bins = np.arange(0, num_cells + 1)

    spike_matrix, neuron_edges, time_edges = np.histogram2d(
        neuron_indices, 
        spike_times,   
        bins=[neuron_bins, time_bins]
    )

    return spike_matrix

def get_params_dict():
    """
    Extracts explicit parameters from the params module.
    Filters out modules and private variables, keeping only uppercase config variables
    that are safe to pickle (scalars, arrays, and Brian2 Quantities).
    """
    params_dict = {}
    for key, val in vars(params).items():
        if key.isupper():
            # Only save specific types. 
            # Brian2 internal structures like DEFAULT_FUNCTIONS cause pickle errors
            if isinstance(val, (int, float, str, bool, np.ndarray, Quantity)):
                params_dict[key] = val
    return params_dict

def save_data(M_N1, M_N2, SM_N1, SM_N2, M_S1_1=None,
              M_S1_2=None, M_S2_1=None, M_S2_2=None, cb_on=True):

    # Save Data, Metadata, and Parameters
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    sim_data = {
        'metadata': {
            'timestamp': datetime.datetime.now().isoformat(),
            'brian2_version': '2.x',
        },
        'params': get_params_dict(),
        'results': {
            't': np.asarray(M_N1.t),
            # POP 1
            'x1': np.asarray(M_N1.x),
            'y1' : np.asarray(M_N1.y),
            'z1': np.asarray(M_N1.z),
            'I_syn_inter_1': np.asarray(M_N1.I_syn_inter),
            'I_syn_intra_1': np.asarray(M_N1.I_syn_intra),
            # POP 2
            'x2': np.asarray(M_N2.x),
            'n2': np.asarray(M_N2.n),
            'I_syn_inter_2': np.asarray(M_N2.I_syn_inter),
            # SPIKES
            'spikes_n1': {'t': np.asarray(SM_N1.t), 'i': np.asarray(SM_N1.i)},
            'spikes_n2': {'t': np.asarray(SM_N2.t), 'i': np.asarray(SM_N2.i)}
        }
    }

    if cb_on:
        # e->e (HR->HR)
        if M_S1_1 is not None:
            sim_data['results'].update({
                'syn_wpre': np.asarray(M_S1_1.Wpre),
                'u':        np.asarray(M_S1_1.u),
                'Ca':       np.asarray(M_S1_1.Ca),
            })
        # e->i (HR->ML)
        if M_S1_2 is not None:
            sim_data['results'].update({
                'S1_2_wpre': np.asarray(M_S1_2.Wpre),
                'S1_2_u':    np.asarray(M_S1_2.u),
                'S1_2_Ca':   np.asarray(M_S1_2.Ca),
            })
        # i->e (ML->HR)
        if M_S2_1 is not None:
            sim_data['results'].update({
                'S2_1_wpre': np.asarray(M_S2_1.Wpre),
                'S2_1_u':    np.asarray(M_S2_1.u),
                'S2_1_Ca':   np.asarray(M_S2_1.Ca),
            })
        # i->i (ML->ML)
        if M_S2_2 is not None:
            sim_data['results'].update({
                'S2_2_wpre': np.asarray(M_S2_2.Wpre),
                'S2_2_u':    np.asarray(M_S2_2.u),
                'S2_2_Ca':   np.asarray(M_S2_2.Ca),
            })


    # Dump to pickle
    filepath = os.path.join(DATA_DIR, OUTPUT_DATA_FILE)
    with open(filepath, 'wb') as f:
        pickle.dump(sim_data, f)
    
    print(f"Simulation data and parameters saved to: {filepath}")

def load_sim_data():
        # Load pickle
    filepath = os.path.join(DATA_DIR, OUTPUT_DATA_FILE)
    with open(filepath, 'rb') as f:
        data = pickle.load(f)

    return data


def cutoff_transient(data, transient, dt):
    # useless function
    if transient > 0:
        start_idx = int(np.ceil(transient/dt)-1)
        if len(data.shape) == 2:
            data = data[:,start_idx:]
        elif len(data.shape) == 1:
            # unique case for time, which is a 1d array
            data = data[start_idx:]
    return data
    

def dump_spikes_to_file(neuron_idx, spike_times):
    """
        dump spike times to a text file for manual correctness checks
    """
    mask = np.where(neuron_idx == 0)
    n0_spikes = spike_times[mask]
    
    np.savetxt('spike_times.txt', n0_spikes, fmt='%f', delimiter=' ')

def dump_array_to_file(arr):
    np.savetxt('r.txt', arr, fmt='%f', delimiter=' ')