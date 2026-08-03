"""
plot_results.py
===============
Generates all figures for the paper from saved .npy files.
Run AFTER run_all.py finishes.

Usage:
    python plot_results.py

Output:
    results/fig4_ce_barchart.png
    results/fig5_learning_curves.png
    results/fig6_mse_loss.png
    results/fig7_heatmap.png
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

os.makedirs('results', exist_ok=True)

DPI = 300
plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size':         9,
    'axes.linewidth':    0.7,
    'grid.linewidth':    0.4,
    'grid.color':        '#BBBBBB',
    'grid.linestyle':    '--',
    'legend.fontsize':   7.5,
    'legend.framealpha': 0.92,
    'legend.edgecolor':  'black',
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'axes.labelsize':    9,
})

# Okabe-Ito palette
OI = dict(black='#000000', orange='#E69F00', sky='#56B4E9',
          green='#009E73', blue='#0072B2', vermil='#D55E00',
          purple='#CC79A7')

CONFIGS = [
    ('DQN+BASELINE',  OI['blue'],   '-',  'o', OI['blue'],  'DQN Baseline'),
    ('DQN+MDWS',      OI['orange'], '-',  's', OI['orange'],'DQN + MDWS'),
    ('DQN+TPDS',      OI['green'],  '-',  '^', OI['green'], 'DQN + TPDS'),
    ('DQN+OPS',       OI['vermil'], '-',  'D', OI['vermil'],'DQN + OPS'),
    ('DDQN+BASELINE', OI['black'],  '--', 'o', 'white',     'DDQN Baseline'),
    ('DDQN+MDWS',     OI['purple'], '--', 's', 'white',     'DDQN + MDWS'),
    ('DDQN+TPDS',     OI['sky'],    '--', '^', 'white',     'DDQN + TPDS'),
    ('DDQN+OPS',      OI['vermil'], '--', 'D', 'white',     'DDQN + OPS'),
]

df = pd.read_csv('results/summary.csv')

# ── FIG 4: CE Bar Chart ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 3.5))
fig.patch.set_facecolor('white')

x       = np.arange(8)
labels  = [r['Config'] for _, r in df.iterrows()]
means   = [r['CE_mean'] for _, r in df.iterrows()]
sds     = [r['CE_sd']   for _, r in df.iterrows()]
colors  = ['#AAAAAA']*4 + ['#555555']*4

ax.bar(x, means, yerr=sds, capsize=4,
       color=colors, edgecolor='black',
       linewidth=0.7, width=0.6,
       error_kw=dict(ecolor='black', elinewidth=0.8, capthick=0.8))

baseline_mean = df[df['Config']=='DQN+BASELINE']['CE_mean'].values[0]
ax.axhline(baseline_mean, color='black',
           linestyle='--', linewidth=0.8, alpha=0.7)

for i, (m, s) in enumerate(zip(means, sds)):
    ax.text(i, m + s + 4, f'{m:.0f}',
            ha='center', va='bottom', fontsize=7)

ax.set_xticks(x)
xlabels = [l.replace('+', '\n+') for l in labels]
ax.set_xticklabels(xlabels, fontsize=7.5)
ax.set_ylabel('Convergence Episode (lower = better)')
ax.set_ylim(0, max(means) * 1.25)
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
ax.set_axisbelow(True)

dqn_p  = mpatches.Patch(fc='#AAAAAA', ec='black', lw=0.7, label='DQN variants')
ddqn_p = mpatches.Patch(fc='#555555', ec='black', lw=0.7, label='DDQN variants')
ax.legend(handles=[dqn_p, ddqn_p], fontsize=7.5, loc='upper right')

for sp in ax.spines.values():
    sp.set_linewidth(0.7)

plt.tight_layout(pad=0.4)
plt.savefig('results/fig4_ce_barchart.png', dpi=DPI, facecolor='white')
plt.close()
print("Saved: fig4_ce_barchart.png")

# ── FIG 5 & 6: Split-panel learning curves and loss curves ───────

def plot_split(data_dict, ylabel, filename, markevery=50):
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.0, 2.8), sharey=True,
        gridspec_kw={'wspace': 0.06})
    fig.patch.set_facecolor('white')

    dqn_cfgs  = [c for c in CONFIGS if c[0].startswith('DQN+')]
    ddqn_cfgs = [c for c in CONFIGS if c[0].startswith('DDQN+')]

    for ax, cfgs, title in [(ax1, dqn_cfgs, '(a) DQN variants'),
                             (ax2, ddqn_cfgs, '(b) DDQN variants')]:
        for key, col, ls, mk, mfc, lbl in cfgs:
            if key not in data_dict:
                continue
            arr  = data_dict[key]             # shape (30, N_EP)
            mean = arr.mean(axis=0)
            x    = np.arange(len(mean))
            ax.plot(x, mean, color=col, linestyle=ls, linewidth=1.0,
                    marker=mk, markersize=3.5, markevery=markevery,
                    markerfacecolor=mfc, markeredgecolor=col,
                    markeredgewidth=0.8, label=lbl)

        ax.set_title(title, fontsize=9, pad=4)
        ax.grid(True)
        ax.tick_params(direction='in', length=3, top=True, right=True)
        for sp in ax.spines.values():
            sp.set_linewidth(0.7)
        leg = ax.legend(fontsize=7, handlelength=2.2,
                        handletextpad=0.4, borderaxespad=0.4,
                        loc='best', labelspacing=0.3)
        leg.get_frame().set_linewidth(0.6)

    ax1.set_ylabel(ylabel)
    ax1.set_xlabel('')
    ax2.set_xlabel('')
    fig.text(0.5, -0.04,
             'Episode' if 'Reward' in ylabel else 'Training Steps',
             ha='center', va='center', fontsize=9)

    plt.tight_layout(pad=0.3)
    plt.savefig(f'results/{filename}', dpi=DPI,
                facecolor='white', bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")


# Load learning curve data
lc_data   = {}
loss_data = {}
for key, *_ in CONFIGS:
    lc_path   = f'results/lc_{key}.npy'
    loss_path = f'results/loss_{key}.npy'
    if os.path.exists(lc_path):
        lc_data[key] = np.load(lc_path)
    if os.path.exists(loss_path):
        loss_data[key] = np.load(loss_path)

plot_split(lc_data,   'Reward (50-episode avg)',
           'fig5_learning_curves.png')
plot_split(loss_data, 'MSE Loss',
           'fig6_mse_loss.png', markevery=40)

# ── FIG 7: Architecture × Shaping Heatmap ────────────────────────
matrix = np.array([
    [df[df['Config']=='DQN+BASELINE']['CE_mean'].values[0],
     df[df['Config']=='DQN+MDWS']['CE_mean'].values[0],
     df[df['Config']=='DQN+TPDS']['CE_mean'].values[0],
     df[df['Config']=='DQN+OPS']['CE_mean'].values[0]],
    [df[df['Config']=='DDQN+BASELINE']['CE_mean'].values[0],
     df[df['Config']=='DDQN+MDWS']['CE_mean'].values[0],
     df[df['Config']=='DDQN+TPDS']['CE_mean'].values[0],
     df[df['Config']=='DDQN+OPS']['CE_mean'].values[0]],
])

fig, ax = plt.subplots(figsize=(4.5, 2.2))
fig.patch.set_facecolor('white')

im = ax.imshow(matrix, cmap='Greys', aspect='auto',
               vmin=matrix.min()*0.9, vmax=matrix.max()*1.05)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05)
cbar.set_label('CE (episodes)', fontsize=7)
cbar.ax.tick_params(labelsize=6)
cbar.ax.invert_yaxis()

row_labels = ['DQN', 'DDQN']
col_labels = ['Baseline', 'MDWS', 'TPDS', 'OPS']

for i in range(2):
    for j in range(4):
        val     = matrix[i, j]
        is_best = (val == matrix.min())
        tc      = 'white' if val < (matrix.min() + matrix.max())/2 else 'black'
        ax.text(j, i, f'{val:.0f}', ha='center', va='center',
                fontsize=10, fontweight='bold' if is_best else 'normal',
                color=tc)
        if is_best:
            rect = plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                  fill=False, edgecolor='black',
                                  linewidth=2.5, zorder=5)
            ax.add_patch(rect)

ax.set_xticks(range(4)); ax.set_xticklabels(col_labels, fontsize=9)
ax.set_yticks(range(2)); ax.set_yticklabels(row_labels, fontsize=9)
ax.set_xlabel('Reward Shaping Function', fontsize=8)
ax.set_ylabel('Architecture', fontsize=8)
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_linewidth(0.7)

plt.tight_layout(pad=0.4)
plt.savefig('results/fig7_heatmap.png', dpi=DPI, facecolor='white')
plt.close()
print("Saved: fig7_heatmap.png")

print("\n✅ All figures saved to results/")