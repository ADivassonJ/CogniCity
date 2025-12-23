import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import norm

# ============================================================
# VALIDACIÓN (OdIN: mean + CI95) vs SIMULACIÓN
# - Medias por arquetipo con incertidumbre comparable
# - Errores: MAE, MAPE/MARE, Bias
# - Cobertura: % arquetipos donde mu_sim cae dentro del IC95 obs
# - Overlap de intervalos: IC_sim ∩ IC_obs
# - z-score y p-value aproximados usando SE_obs inferido del IC95 obs
# - (Descriptivo) Cuantiles/SD/IQR SOLO de simulación (sin comparar)
# ============================================================

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
SIM_FILE = os.path.expanduser("~/Desktop/Kanaleneiland_schedule_citizen.xlsx")

# Si tienes réplicas reales (recomendado), descomenta y ajusta:
# SIM_REPLICATE_GLOB = os.path.expanduser("~/Desktop/sim_outputs/seed_*/Kanaleneiland_schedule_citizen.xlsx")
SIM_REPLICATE_GLOB = None

SEED = 42
R_SIM_BOOT = 50            # bootstrap sobre agentes si no hay réplicas reales
ALPHA = 0.05               # 95% CI
QUANTILES = [0.10, 0.50, 0.90]

rng = np.random.default_rng(SEED)

# -----------------------------
# OdIN: SOLO (value, lower95, upper95)
# (tal cual la tabla que has pegado)
# -----------------------------
ODIN_CI_TABLE = {
    "c_arch_0": {
        "amount":   {"mu": 2.79,   "low": 2.69,   "high": 2.88},
        "distance": {"mu": 48.86,  "low": 46.10,  "high": 51.63},
        "time":     {"mu": 86.12,  "low": 82.49,  "high": 89.75},
    },
    "c_arch_1": {
        "amount":   {"mu": 3.26,   "low": 3.12,   "high": 3.39},
        "distance": {"mu": 43.11,  "low": 39.92,  "high": 46.29},
        "time":     {"mu": 87.84,  "low": 83.42,  "high": 92.26},
    },
    "c_arch_2": {
        "amount":   {"mu": 2.82,   "low": 2.635,  "high": 3.00},
        "distance": {"mu": 18.805, "low": 15.505, "high": 22.11},
        "time":     {"mu": 61.335, "low": 55.105, "high": 67.565},
    },
    "c_arch_3": {
        "amount":   {"mu": 2.35,   "low": 2.205,  "high": 2.505},
        "distance": {"mu": 26.19,  "low": 22.645, "high": 29.74},
        "time":     {"mu": 66.54,  "low": 60.915, "high": 72.16},
    },
    "c_arch_4": {
        "amount":   {"mu": 2.69,   "low": 2.56,   "high": 2.815},
        "distance": {"mu": 40.78,  "low": 37.215, "high": 44.345},
        "time":     {"mu": 84.435, "low": 79.29,  "high": 89.59},
    },
}

ARCH_ORDER = list(ODIN_CI_TABLE.keys())

# ------------------------------------------------------------
# IMPORTANTE: Selecciona qué variable estás validando
# - "time" es el caso de commuting time (min/día)
# - "distance" si tienes km/día en la simulación
# - "amount" si tienes nº de trips/día
# ------------------------------------------------------------
TARGET = "time"

# ------------------------------------------------------------
# IMPORTANTE: Define cómo calcular en la simulación la variable TARGET
# En tu Excel actual solo tenemos "in/out" y "archetype".
# Aquí mantenemos tu definición previa:
#   simulated_time = 1440 - sum(spended)
# Si esto NO es commuting time, cambia la función compute_target_per_agent()
# ------------------------------------------------------------

def compute_target_per_agent(df_activities: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Recibe df de actividades (nivel fila) y devuelve df por agente con:
      agent, archetype, value
    """
    # Asegurar numéricos si existen
    for c in ["in", "out"]:
        if c in df_activities.columns:
            df_activities[c] = pd.to_numeric(df_activities[c], errors="coerce")

    # Calcular spended (min)
    if "in" in df_activities.columns and "out" in df_activities.columns:
        df_activities["spended"] = df_activities["out"] - df_activities["in"]
        df_activities.loc[(df_activities["spended"] < 0) | (df_activities["spended"] > 1440), "spended"] = np.nan
    else:
        raise ValueError("No existen columnas 'in' y 'out' necesarias para calcular spended.")

    # Agregar por agente (un día)
    df_agent = (
        df_activities.groupby("agent", observed=True)
        .agg(
            archetype=("archetype", "first"),
            suma_spended=("spended", "sum")
        )
        .reset_index()
    )

    # Definición actual (tu pipeline):
    # "time" = 1440 - suma_spended
    if target == "time":
        df_agent["value"] = 1440 - df_agent["suma_spended"]
    else:
        # Si más adelante quieres distance/amount necesitarás columnas específicas.
        raise ValueError(
            f"TARGET='{target}' no está implementado con este Excel. "
            "Necesitas columnas en el input para distance/amount o definir el cálculo."
        )

    df_agent["archetype"] = df_agent["archetype"].astype(str).str.strip()
    df_agent.loc[(df_agent["value"] < 0) | (df_agent["value"] > 1440), "value"] = np.nan

    return df_agent[["agent", "archetype", "value"]].dropna()

def load_sim_from_excel(path: str, target: str) -> pd.DataFrame:
    df = pd.read_excel(path, decimal=",", engine="openpyxl")
    if "archetype" not in df.columns or "agent" not in df.columns:
        raise ValueError("El Excel debe tener columnas 'agent' y 'archetype'.")
    return compute_target_per_agent(df, target=target)

def get_sim_replicates():
    if SIM_REPLICATE_GLOB:
        paths = sorted(glob.glob(SIM_REPLICATE_GLOB))
        if not paths:
            raise FileNotFoundError(f"No se encontraron ficheros con el patrón: {SIM_REPLICATE_GLOB}")
        reps = [load_sim_from_excel(p, target=TARGET) for p in paths]
        return reps, paths
    else:
        df0 = load_sim_from_excel(SIM_FILE, target=TARGET)
        return [df0], [SIM_FILE]

# -----------------------------
# Estadísticos
# -----------------------------
def iqr(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan
    return float(np.quantile(x, 0.75) - np.quantile(x, 0.25))

def infer_se_from_ci95(low, high):
    # SE ≈ (high - low) / (2*1.96)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return np.nan
    return float((high - low) / (2.0 * 1.96))

def mu_ci_from_replicates(sim_reps):
    out = {}
    for arch in ARCH_ORDER:
        mus = []
        for rep in sim_reps:
            x = rep.loc[rep["archetype"] == arch, "value"].to_numpy()
            x = x[np.isfinite(x)]
            mus.append(float(np.mean(x)) if len(x) else np.nan)
        mus = np.asarray(mus, dtype=float)
        mus = mus[np.isfinite(mus)]
        if len(mus) < 2:
            out[arch] = (np.nan, np.nan, np.nan, np.nan)
        else:
            mu = float(np.mean(mus))
            lo = float(np.quantile(mus, 0.025))
            hi = float(np.quantile(mus, 0.975))
            se = float(np.std(mus, ddof=1))
            out[arch] = (mu, lo, hi, se)
    return out

def mu_ci_from_bootstrap_agents(sim_df, R=50):
    out = {}
    for arch in ARCH_ORDER:
        x = sim_df.loc[sim_df["archetype"] == arch, "value"].to_numpy()
        x = x[np.isfinite(x)]
        if len(x) < 2:
            out[arch] = (np.nan, np.nan, np.nan, np.nan)
            continue
        n = len(x)
        boots = np.empty(R, dtype=float)
        for r in range(R):
            idx = rng.choice(n, size=n, replace=True)
            boots[r] = float(np.mean(x[idx]))
        mu = float(np.mean(x))
        lo = float(np.quantile(boots, 0.025))
        hi = float(np.quantile(boots, 0.975))
        se = float(np.std(boots, ddof=1))
        out[arch] = (mu, lo, hi, se)
    return out

def fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    return f"{x:.{nd}f}".replace(".", ",")

def fmt_pct(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    return f"{x:.{nd}f}%".replace(".", ",")

def fmt_bool(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    return "yes" if bool(x) else "no"

# ============================================================
# EJECUCIÓN
# ============================================================

sim_reps, sim_paths = get_sim_replicates()

if len(sim_reps) >= 2:
    sim_mu_ci = mu_ci_from_replicates(sim_reps)
    sim_unc_method = "between-replicates"
else:
    sim_mu_ci = mu_ci_from_bootstrap_agents(sim_reps[0], R=R_SIM_BOOT)
    sim_unc_method = "bootstrap-agents (fallback)"

# -----------------------------
# TABLA 1: Medias + IC + errores + coverage/overlap + z-test aproximado
# -----------------------------
rows = []
for arch in ARCH_ORDER:
    mu_sim, lo_sim, hi_sim, se_sim = sim_mu_ci.get(arch, (np.nan, np.nan, np.nan, np.nan))

    mu_obs = ODIN_CI_TABLE[arch][TARGET]["mu"]
    lo_obs = ODIN_CI_TABLE[arch][TARGET]["low"]
    hi_obs = ODIN_CI_TABLE[arch][TARGET]["high"]

    se_obs = infer_se_from_ci95(lo_obs, hi_obs)

    err_min = mu_sim - mu_obs
    err_rel = (err_min / mu_obs * 100.0) if np.isfinite(mu_obs) and mu_obs != 0 else np.nan

    # Coverage: mu_sim dentro IC95 obs
    in_obs_ci = (mu_sim >= lo_obs) and (mu_sim <= hi_obs) if all(np.isfinite([mu_sim, lo_obs, hi_obs])) else np.nan

    # Overlap de intervalos
    overlap = (min(hi_sim, hi_obs) - max(lo_sim, lo_obs)) if all(np.isfinite([lo_sim, hi_sim, lo_obs, hi_obs])) else np.nan
    overlap_yes = (overlap >= 0) if np.isfinite(overlap) else np.nan

    # z-score aproximado usando SE_obs inferido del IC95 de la media
    z = (mu_sim - mu_obs) / se_obs if np.isfinite(se_obs) and se_obs > 0 else np.nan
    p2 = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan

    rows.append({
        "Archetype": arch,
        "mu_sim": mu_sim, "ci95_sim_low": lo_sim, "ci95_sim_high": hi_sim, "se_sim": se_sim,
        "mu_obs": mu_obs, "ci95_obs_low": lo_obs, "ci95_obs_high": hi_obs, "se_obs": se_obs,
        "error_min": err_min, "error_rel_%": err_rel,
        "in_obs_ci95": in_obs_ci,
        "ci_overlap": overlap, "ci_overlap_yes": overlap_yes,
        "z_using_se_obs": z, "p_2sided_using_se_obs": p2
    })

df_means = pd.DataFrame(rows).set_index("Archetype")

MAE = float(np.nanmean(np.abs(df_means["error_min"])))
MAPE = float(np.nanmean(np.abs(df_means["error_rel_%"])))
BIAS = float(np.nanmean(df_means["error_min"]))
COVERAGE = float(np.nanmean(df_means["in_obs_ci95"]))
OVERLAP_RATE = float(np.nanmean(df_means["ci_overlap_yes"]))

# Formateo
df_means_fmt = df_means.copy()
for c in ["mu_sim","ci95_sim_low","ci95_sim_high","se_sim","mu_obs","ci95_obs_low","ci95_obs_high","se_obs",
          "error_min","ci_overlap","z_using_se_obs","p_2sided_using_se_obs"]:
    df_means_fmt[c] = df_means_fmt[c].map(lambda v: fmt(v, 3))
df_means_fmt["error_rel_%"] = df_means_fmt["error_rel_%"].map(lambda v: fmt_pct(v, 2))
df_means_fmt["in_obs_ci95"] = df_means_fmt["in_obs_ci95"].map(fmt_bool)
df_means_fmt["ci_overlap_yes"] = df_means_fmt["ci_overlap_yes"].map(fmt_bool)

print("\n====================")
print(f"TABLA 1: Medias + IC (TARGET='{TARGET}')")
print("====================")
print(df_means_fmt)

print("\nResumen global (medias)")
print("Método incertidumbre simulación:", sim_unc_method)
print("MAE (min):", fmt(MAE, 3))
print("MAPE/MARE (abs, %):", fmt_pct(MAPE, 2))
print("Bias (min):", fmt(BIAS, 3))
print("Coverage (mu_sim in IC95 obs):", fmt_pct(COVERAGE * 100, 1))
print("IC overlap rate (IC_sim overlaps IC_obs):", fmt_pct(OVERLAP_RATE * 100, 1))

# -----------------------------
# TABLA 2: Distribución SOLO SIM (cuantiles + dispersión)
# (No se compara con obs porque OdIN no da cuantiles/ECDF)
# -----------------------------
concat = pd.concat(sim_reps, ignore_index=True)

rows = []
for arch in ARCH_ORDER:
    x = concat.loc[concat["archetype"] == arch, "value"].dropna().to_numpy()
    x = x[np.isfinite(x)]

    qsim = {q: float(np.quantile(x, q)) if len(x) else np.nan for q in QUANTILES}
    rows.append({
        "Archetype": arch,
        "Q10_sim": qsim[0.10],
        "Q50_sim": qsim[0.50],
        "Q90_sim": qsim[0.90],
        "SD_sim": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
        "IQR_sim": iqr(x),
        "N_sim": int(len(x)),
    })

df_sim_dist = pd.DataFrame(rows).set_index("Archetype")

df_sim_dist_fmt = df_sim_dist.copy()
for c in ["Q10_sim","Q50_sim","Q90_sim","SD_sim","IQR_sim"]:
    df_sim_dist_fmt[c] = df_sim_dist_fmt[c].map(lambda v: fmt(v, 3))
df_sim_dist_fmt["N_sim"] = df_sim_dist_fmt["N_sim"].astype(int)

print("\n====================")
print("TABLA 2: Distribución (solo simulación): cuantiles y dispersión")
print("====================")
print(df_sim_dist_fmt)

# -----------------------------
# EXPORT
# -----------------------------
out_xlsx = os.path.expanduser(f"~/Desktop/validation_{TARGET}_odin_ci_only.xlsx")
with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
    df_means.to_excel(writer, sheet_name="means_ci_errors")
    df_sim_dist.to_excel(writer, sheet_name="sim_quantiles_disp")
print("\nGuardado:", out_xlsx)
