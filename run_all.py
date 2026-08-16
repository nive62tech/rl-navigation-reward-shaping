# run_all.py
import os, random
import numpy as np
import pandas as pd
from scipy import stats
from train import run_one

os.makedirs('results', exist_ok=True)

CONFIGS    = [('dqn','baseline'),('dqn','mdws'),('dqn','tpds'),('dqn','ops'),
              ('ddqn','baseline'),('ddqn','mdws'),('ddqn','tpds'),('ddqn','ops')]
N_SEEDS    = 30
N_EPISODES = 600

ce_all={}; mse_all={}; scr_all={}

print("="*50)
print(f"Running {len(CONFIGS)} configs x {N_SEEDS} seeds")
print("="*50)

for arch, shaping in CONFIGS:
    key = f"{arch.upper()}+{shaping.upper()}"
    print(f"\n-- {key} --")
    ces,mses,scrs,lcs,losses=[],[],[],[],[]

    for seed in range(1, N_SEEDS+1):
        ce,lc,loss,mse,scr = run_one(arch=arch,shaping=shaping,
                                      seed=seed,n_episodes=N_EPISODES)
        ces.append(ce); mses.append(mse); scrs.append(scr)
        lcs.append(lc); losses.append(loss)
        print(f"  seed {seed:2d} | CE={ce:4d} | MSE={mse:.4f} | SCR={scr:.2f}")

    ce_all[key]=ces; mse_all[key]=mses; scr_all[key]=scrs
    np.save(f'results/lc_{key}.npy',   np.array(lcs))
    np.save(f'results/loss_{key}.npy', np.array(losses))
    print(f"  -> CE {np.mean(ces):.2f} +/- {np.std(ces,ddof=1):.2f}")

pd.DataFrame(ce_all).to_csv('results/ce_raw.csv',   index=False)
pd.DataFrame(mse_all).to_csv('results/mse_raw.csv', index=False)
pd.DataFrame(scr_all).to_csv('results/scr_raw.csv', index=False)

def cohens_d(a,b):
    na,nb=len(a),len(b)
    p=np.sqrt(((na-1)*np.std(a,ddof=1)**2+(nb-1)*np.std(b,ddof=1)**2)/(na+nb-2))
    return abs(np.mean(a)-np.mean(b))/p if p>0 else 0.0

base_ce   = ce_all['DQN+BASELINE']
THRESHOLD = 0.05/(len(CONFIGS)-1)
rows=[]

for key in ce_all:
    ces=ce_all[key]; mses=mse_all[key]; scrs=scr_all[key]
    delta=(np.mean(ces)-np.mean(base_ce))/np.mean(base_ce)*100
    if key!='DQN+BASELINE':
        u,p=stats.mannwhitneyu(ces,base_ce,alternative='less')
        sig='Yes' if p<THRESHOLD else 'No'
        d=cohens_d(base_ce,ces)
    else:
        u=p=d='-'; sig='-'
    rows.append(dict(
        Config=key,
        CE_mean=round(np.mean(ces),2), CE_sd=round(np.std(ces,ddof=1),2),
        Delta=round(delta,2),
        MSE_mean=round(np.mean(mses),4), MSE_sd=round(np.std(mses,ddof=1),4),
        SCR_mean=round(np.nanmean(scrs),2), SCR_sd=round(np.nanstd(scrs,ddof=1),2),
        U_stat=round(float(u),1) if u!='-' else '-',
        p_value=round(float(p),5) if p!='-' else '-',
        Sig=sig,
        Cohen_d=round(float(d),3) if d!='-' else '-',
    ))

df=pd.DataFrame(rows)
df.to_csv('results/summary.csv',index=False)

print("\n"+"="*50)
print("TABLE II -- Convergence Episodes")
print("="*50)
for _,r in df.iterrows():
    d=f"{r['Delta']:.2f}%" if r['Delta']!=0 else "-- reference"
    print(f"  {r['Config']:<22} {r['CE_mean']:.2f} +/- {r['CE_sd']:.2f}   {d}")

print("\n"+"="*50)
print("TABLE III -- MSE and SCR")
print("="*50)
for _,r in df.iterrows():
    print(f"  {r['Config']:<22} MSE={r['MSE_mean']:.4f}+/-{r['MSE_sd']:.4f}"
          f"  SCR={r['SCR_mean']:.2f}+/-{r['SCR_sd']:.2f}")

print("\n"+"="*50)
print(f"TABLE IV -- Stats (threshold={THRESHOLD:.4f})")
print("="*50)
for _,r in df.iterrows():
    if r['Config']=='DQN+BASELINE': continue
    print(f"  {r['Config']:<22} U={r['U_stat']}  p={r['p_value']}"
          f"  Sig={r['Sig']}  d={r['Cohen_d']}")

# Ablation
print("\n"+"="*50)
print("ABLATION -- alpha sensitivity")
print("="*50)
from shaping import phi_mdws, SHAPING_MAP
abl=[]
for alpha in [0.1,0.25,0.5,0.75,1.0]:
    def phi_a(env,t=None,_a=alpha):
        mx=(env.N-1)*2
        return -_a*(env.manhattan_to_goal()/mx)
    SHAPING_MAP['mdws']=phi_a
    ces=[run_one('dqn','mdws',seed=s,n_episodes=N_EPISODES)[0]
         for s in range(1,11)]
    m,s=np.mean(ces),np.std(ces,ddof=1)
    print(f"  alpha={alpha:.2f}  CE={m:.2f} +/- {s:.2f}")
    abl.append(dict(alpha=alpha,CE_mean=round(m,2),CE_sd=round(s,2)))
SHAPING_MAP['mdws']=phi_mdws
pd.DataFrame(abl).to_csv('results/ablation.csv',index=False)
print("\n✅ Done. results/ folder is ready.")