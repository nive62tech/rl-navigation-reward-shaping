"""
run_all.py
==========
Runs all 8 configurations x 30 seeds.
Saves results to results/ folder.
Run: python run_all.py
"""
import os
import random
import numpy as np
import pandas as pd
from scipy import stats
from train import run_one

os.makedirs('results', exist_ok=True)

CONFIGS = [
    ('dqn',  'baseline'),
    ('dqn',  'mdws'),
    ('dqn',  'tpds'),
    ('dqn',  'ops'),
    ('ddqn', 'baseline'),
    ('ddqn', 'mdws'),
    ('ddqn', 'tpds'),
    ('ddqn', 'ops'),
]

N_SEEDS    = 30
N_EPISODES = 600

ce_all   = {}
mse_all  = {}
scr_all  = {}
lc_all   = {}
loss_all = {}

print("=" * 55)
print(f"Running {len(CONFIGS)} configs x {N_SEEDS} seeds")
print("=" * 55)

for arch, shaping in CONFIGS:
    key = f"{arch.upper()}+{shaping.upper()}"
    print(f"\n-- {key} --")
    ces, mses, scrs, lcs, losses = [], [], [], [], []

    for seed in range(1, N_SEEDS + 1):
        ce, lc, loss, mse, scr = run_one(
            arch=arch, shaping=shaping,
            seed=seed, n_episodes=N_EPISODES)
        ces.append(ce)
        mses.append(mse)
        scrs.append(scr)
        lcs.append(lc)
        losses.append(loss)
        print(f"  seed {seed:2d} | CE={ce:4d} | MSE={mse:.4f} | SCR={scr:.2f}")

    ce_all[key]   = ces
    mse_all[key]  = mses
    scr_all[key]  = scrs
    lc_all[key]   = np.array(lcs)
    loss_all[key] = np.array(losses)

    np.save(f'results/lc_{key}.npy',   np.array(lcs))
    np.save(f'results/loss_{key}.npy', np.array(losses))

    print(f"  -> CE  {np.mean(ces):.2f} +/- {np.std(ces, ddof=1):.2f}")
    print(f"  -> MSE {np.mean(mses):.4f} +/- {np.std(mses, ddof=1):.4f}")
    print(f"  -> SCR {np.nanmean(scrs):.2f} +/- {np.nanstd(scrs, ddof=1):.2f}")

pd.DataFrame(ce_all).to_csv('results/ce_raw.csv',   index=False)
pd.DataFrame(mse_all).to_csv('results/mse_raw.csv', index=False)
pd.DataFrame(scr_all).to_csv('results/scr_raw.csv', index=False)


def cohens_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(
        ((na-1)*np.std(a,ddof=1)**2 + (nb-1)*np.std(b,ddof=1)**2)
        / (na+nb-2))
    return abs(np.mean(a)-np.mean(b)) / pooled if pooled > 0 else 0.0

baseline_ce   = ce_all['DQN+BASELINE']
N_COMPARISONS = len(CONFIGS) - 1
THRESHOLD     = 0.05 / N_COMPARISONS

rows = []
for key in ce_all:
    ces  = ce_all[key]
    mses = mse_all[key]
    scrs = scr_all[key]

    ce_mean  = np.mean(ces)
    ce_sd    = np.std(ces, ddof=1)
    mse_mean = np.mean(mses)
    mse_sd   = np.std(mses, ddof=1)
    scr_mean = np.nanmean(scrs)
    scr_sd   = np.nanstd(scrs, ddof=1)
    delta    = (ce_mean - np.mean(baseline_ce)) / np.mean(baseline_ce) * 100

    if key != 'DQN+BASELINE':
        u_stat, p_val = stats.mannwhitneyu(
            ces, baseline_ce, alternative='less')
        sig = 'Yes' if p_val < THRESHOLD else 'No'
        d   = cohens_d(baseline_ce, ces)
    else:
        u_stat = p_val = d = '-'
        sig = '-'

    rows.append(dict(
        Config   = key,
        CE_mean  = round(ce_mean,  2),
        CE_sd    = round(ce_sd,    2),
        Delta    = round(delta,    2),
        MSE_mean = round(mse_mean, 4),
        MSE_sd   = round(mse_sd,   4),
        SCR_mean = round(scr_mean, 2),
        SCR_sd   = round(scr_sd,   2),
        U_stat   = round(float(u_stat), 1) if u_stat != '-' else '-',
        p_value  = round(float(p_val),  5) if p_val  != '-' else '-',
        Sig      = sig,
        Cohen_d  = round(float(d), 3)      if d      != '-' else '-',
    ))

df = pd.DataFrame(rows)
df.to_csv('results/summary.csv', index=False)

print("\n" + "=" * 55)
print("TABLE II -- Convergence Episodes")
print("=" * 55)
for _, r in df.iterrows():
    d = f"{r['Delta']:.2f}%" if r['Delta'] != 0 else "-- reference"
    print(f"  {r['Config']:<22} {r['CE_mean']:.2f} +/- {r['CE_sd']:.2f}   {d}")

print("\n" + "=" * 55)
print("TABLE III -- MSE and SCR")
print("=" * 55)
for _, r in df.iterrows():
    print(f"  {r['Config']:<22} MSE={r['MSE_mean']:.4f}+/-{r['MSE_sd']:.4f}"
          f"  SCR={r['SCR_mean']:.2f}+/-{r['SCR_sd']:.2f}")

print("\n" + "=" * 55)
print(f"TABLE IV -- Statistical Tests (threshold={THRESHOLD:.4f})")
print("=" * 55)
for _, r in df.iterrows():
    if r['Config'] == 'DQN+BASELINE':
        continue
    print(f"  {r['Config']:<22} U={r['U_stat']}  "
          f"p={r['p_value']}  Sig={r['Sig']}  d={r['Cohen_d']}")

# Ablation
print("\n" + "=" * 55)
print("ABLATION -- alpha sensitivity (MDWS, DQN, 10 seeds)")
print("=" * 55)

from shaping import phi_mdws, SHAPING_MAP

ALPHAS = [0.1, 0.25, 0.5, 0.75, 1.0]
abl_rows = []

for alpha in ALPHAS:
    def phi_a(env, t=None, _alpha=alpha):
        return -_alpha * env.manhattan_to_goal()

    SHAPING_MAP['mdws'] = phi_a
    ces = []
    for seed in range(1, 11):
        ce, *_ = run_one('dqn', 'mdws', seed=seed, n_episodes=N_EPISODES)
        ces.append(ce)
    m, s = np.mean(ces), np.std(ces, ddof=1)
    print(f"  alpha={alpha:.2f}  CE={m:.2f} +/- {s:.2f}")
    abl_rows.append(dict(alpha=alpha, CE_mean=round(m,2), CE_sd=round(s,2)))

SHAPING_MAP['mdws'] = phi_mdws
pd.DataFrame(abl_rows).to_csv('results/ablation.csv', index=False)

print("\n✅ All done. Download results/ folder.")