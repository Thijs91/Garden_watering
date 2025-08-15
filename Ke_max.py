import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize
from tqdm import trange

# === Instelbare parameters ===
DEPTH_EFF_CM = 12           # Sensor diepte in cm (12cm = 120mm)
KE_MAX_INIT = 0.15          # Startwaarde voor optimalisatie
SM_DRY = 30                 # Droogdrempel (%)
SM_WET = 47                 # Natdrempel (%)
BOOTSTRAP_N = 1000          # Aantal bootstrap herhalingen
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# === CSV inladen ===
# Vereiste kolommen: timestamp, soil_moisture, et0, etc_obs, rain
df = pd.read_csv("data.csv", parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Filter droge periodes (geen regen)
df = df[df["rain"] == 0].copy()

# === Functie om Ke te berekenen ===
def calc_ke(sm, ke_max):
    """Lineaire Ke van SM_DRY naar SM_WET met cap ke_max."""
    return np.where(sm <= SM_DRY, 0,
           np.where(sm >= SM_WET, ke_max,
                    ke_max * (sm - SM_DRY) / (SM_WET - SM_DRY)))

# === Doelfunctie voor optimalisatie ===
def objective(ke_max, df_input):
    ke_series = calc_ke(df_input["soil_moisture"], ke_max)
    df_input["etc_model"] = df_input["et0"] * (1 + ke_series)
    return mean_squared_error(df_input["etc_obs"], df_input["etc_model"])

# === Optimalisatie van ke_max ===
result = minimize(lambda x: objective(x[0], df), [KE_MAX_INIT], bounds=[(0, 0.4)])
ke_best = result.x[0]

# === Statistieken berekenen ===
df["ke_model"] = calc_ke(df["soil_moisture"], ke_best)
df["etc_model"] = df["et0"] * (1 + df["ke_model"])

obs = df["etc_obs"].values
mod = df["etc_model"].values

mse = mean_squared_error(obs, mod)
rmse = np.sqrt(mse)
mae = mean_absolute_error(obs, mod)
mape = np.mean(np.abs((obs - mod) / obs)) * 100
bias = np.mean(mod - obs)
r2 = r2_score(obs, mod)
nse = 1 - np.sum((obs - mod)**2) / np.sum((obs - np.mean(obs))**2)
corr_p, _ = pearsonr(obs, mod)
corr_s, _ = spearmanr(obs, mod)

# === Bland–Altman data ===
mean_vals = (obs + mod) / 2
diff_vals = mod - obs
mean_diff = np.mean(diff_vals)
sd_diff = np.std(diff_vals)

# === Bootstrap CI voor ke_max ===
bootstrap_ke = []
for _ in trange(BOOTSTRAP_N, desc="Bootstrap"):
    sample_idx = np.random.choice(len(df), size=len(df), replace=True)
    df_sample = df.iloc[sample_idx].copy()
    res = minimize(lambda x: objective(x[0], df_sample), [KE_MAX_INIT], bounds=[(0, 0.4)])
    bootstrap_ke.append(res.x[0])

ci_low = np.percentile(bootstrap_ke, 2.5)
ci_high = np.percentile(bootstrap_ke, 97.5)

# === Resultaten printen ===
print("\n=== Resultaten optimalisatie ===")
print(f"Beste ke_max: {ke_best:.3f}")
print(f"95% CI bootstrap: [{ci_low:.3f}, {ci_high:.3f}]")
print(f"RMSE: {rmse:.3f} mm")
print(f"MAE: {mae:.3f} mm")
print(f"MAPE: {mape:.2f} %")
print(f"Bias: {bias:.3f} mm")
print(f"R²: {r2:.3f}")
print(f"NSE: {nse:.3f}")
print(f"Pearson r: {corr_p:.3f}")
print(f"Spearman rho: {corr_s:.3f}")

# === Grafieken ===

# 1. Observed vs Model
plt.figure(figsize=(6,6))
plt.scatter(obs, mod, alpha=0.6)
plt.plot([min(obs), max(obs)], [min(obs), max(obs)], 'r--', label="1:1 lijn")
plt.xlabel("ETc Observed (mm)")
plt.ylabel("ETc Model (mm)")
plt.title("Observed vs Model ETc")
plt.legend()
plt.grid(True)

# 2. Ke en bodemvocht over tijd
fig, ax1 = plt.subplots(figsize=(10,5))
ax1.set_xlabel("Tijd")
ax1.set_ylabel("Bodemvocht (%)", color="tab:blue")
ax1.plot(df["timestamp"], df["soil_moisture"], color="tab:blue")
ax2 = ax1.twinx()
ax2.set_ylabel("Ke", color="tab:orange")
ax2.plot(df["timestamp"], df["ke_model"], color="tab:orange")
plt.title("Ke en bodemvocht over tijd")
fig.tight_layout()

# 3. ETc tijdserie
plt.figure(figsize=(10,5))
plt.plot(df["timestamp"], obs, label="Observed")
plt.plot(df["timestamp"], mod, label="Model")
plt.xlabel("Tijd")
plt.ylabel("ETc (mm)")
plt.title("ETc Observed vs Model")
plt.legend()
plt.grid(True)

# 4. Bland–Altman plot
plt.figure(figsize=(6,6))
plt.scatter(mean_vals, diff_vals, alpha=0.6)
plt.axhline(mean_diff, color='red', linestyle='--', label=f"Mean diff: {mean_diff:.3f}")
plt.axhline(mean_diff + 1.96*sd_diff, color='gray', linestyle='--', label="+1.96 SD")
plt.axhline(mean_diff - 1.96*sd_diff, color='gray', linestyle='--', label="-1.96 SD")
plt.xlabel("Gemiddelde ETc (mm)")
plt.ylabel("Verschil Model - Observed (mm)")
plt.title("Bland–Altman plot")
plt.legend()
plt.grid(True)

plt.show()
