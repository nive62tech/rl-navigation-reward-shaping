# ================================================================
# COLAB NOTEBOOK — Run this top to bottom
# Each cell is separated by # ── CELL ──
# ================================================================

# ── CELL 1: Clone repo and install ──────────────────────────────
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
%cd YOUR_REPO_NAME
!pip install torch numpy pandas scipy matplotlib gymnasium -q

# ── CELL 2: Test the environment works ──────────────────────────
from env import GridWorld

env  = GridWorld(seed=1)
obs  = env.reset()
print(f"Obs dimension : {obs.shape}")
print(f"Obstacles     : {len(env.obstacles)}")

obs2, r, done, info = env.step(1)
print(f"Step result   : reward={r}, done={done}, info={info}")
print("✅ Environment works")

# ── CELL 3: Test one training run ───────────────────────────────
from train import run_one

print("Running one test episode (DQN + TPDS, seed=1) ...")
ce, lc, loss, mse, scr = run_one('dqn', 'tpds', seed=1, n_episodes=50)
print(f"CE={ce}  MSE={mse:.4f}  SCR={scr:.2f}")
print("✅ Training works")

# ── CELL 4: Run ALL experiments (takes 3-6 hours) ───────────────
# This runs 8 configs × 30 seeds and saves everything to results/
# DO NOT close the tab while this runs

!python run_all.py

# ── CELL 5: Generate all figures ────────────────────────────────
!python plot_results.py

# ── CELL 6: Download all results ────────────────────────────────
import shutil
from google.colab import files

# Zip the results folder
shutil.make_archive('results', 'zip', 'results')
files.download('results.zip')
print("✅ Downloaded results.zip")
# Inside results.zip you will find:
#   summary.csv          ← paste numbers into paper
#   ablation.csv         ← ablation table
#   fig4_ce_barchart.png
#   fig5_learning_curves.png
#   fig6_mse_loss.png
#   fig7_heatmap.png
#   ce_raw.csv / mse_raw.csv / scr_raw.csv  ← raw data
