from __future__ import print_function, division
import numpy as np
import os

SAVE_DIR = '/Users/rkp/Dropbox/Repositories/sta_practice/'

T = 50000
T_FILTER = 20
TAU_FILTER = 4
G_NL = 3
TH_NL = 3


def phi(s0):
    return 1 / (1 + np.exp(-G_NL * (s0 - TH_NL)))

t_filter = np.arange(T_FILTER)
stim_filter = np.exp(-t_filter / TAU_FILTER)[::-1]

stim = np.random.normal(0, 1, T)
spikes = np.zeros(stim.shape)

for ts in range(T_FILTER, T - 1):
    windowed_stim = stim[ts-T_FILTER:ts]
    s0 = np.dot(windowed_stim, stim_filter)
    p_spike = phi(s0)
    spike = int(np.random.rand() < p_spike)
    spikes[ts + 1] = spike

np.save(os.path.join(SAVE_DIR, 'stim.npy'), stim)
np.save(os.path.join(SAVE_DIR, 'spikes.npy'), spikes)

print('Number of spikes: {}'.format(spikes.sum()))