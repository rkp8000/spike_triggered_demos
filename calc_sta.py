from __future__ import print_function, division
import matplotlib.pyplot as plt
import numpy as np
import os

LOAD_DIR = '/Users/rkp/Dropbox/Repositories/sta_practice/'
STIM_FILE = 'stim.npy'
SPIKES_FILE = 'spikes.npy'

N_BINS = 50

T_STA = 20


stim = np.load(STIM_FILE)
spikes = np.load(SPIKES_FILE)

T = len(stim)

ste = []

for ts in range(T_STA, T - 1):
    if spikes[ts + 1]:
        ste.append(stim[ts-T_STA:ts])

ste = np.array(ste)
sta = ste.mean(0)

print('sta calculated')

prior = []
posterior = []

for ts in range(T_STA, T - 1):
    stim_window = stim[ts - T_STA:ts]
    s0 = stim_window.dot(sta)
    prior.append(s0)
    if spikes[ts + 1]:
        posterior.append(s0)

cts_prior, bins = np.histogram(prior, bins=N_BINS, normed=True)
cts_posterior, bins = np.histogram(posterior, bins=bins, normed=True)
bincs = .5 * (bins[:-1] + bins[1:])

p_spike = spikes.mean()

nl = p_spike * cts_posterior / cts_prior

fig, axs = plt.subplots(1, 2, facecolor='white', figsize=(10, 6))
axs[0].plot(np.arange(-T_STA, 0), sta)
line0, = axs[1].plot(bincs, cts_prior, c='b')
line1, = axs[1].plot(bincs, cts_posterior, c='r')

ax1_twin = axs[1].twinx()
line2, = ax1_twin.plot(bincs, nl, c='k')
ax1_twin.set_ylim(0, 1.1)
ax1_twin.set_ylabel('nonlinearity')

axs[0].set_xlabel('time steps relative to spike')
axs[0].set_ylabel('sta value')
axs[0].set_title('STA')

axs[1].set_xlabel('s0')
axs[1].set_ylabel('probability')
axs[1].legend([line0, line1, line2], ['p(s0)', 'p(s0|spike)', 'nonlinearity'], loc='best')
axs[1].set_title('prior, posterior, and nonlinearity')

plt.show()