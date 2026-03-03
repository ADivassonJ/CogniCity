import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# -------------------------------
# CONSTANTES
# -------------------------------
MJ_TO_KWH = 1.0 / 3.6
fig_width = 368 / 25.4
fig_height = 78 / 25.4
COLORS = ["#76a5af", "#6aa84f", "#e69138", "#8e7cc3", "#0c343d"]

# -------------------------------
# FUNCIONES BASE
# -------------------------------

def parse_hour(timestr: str) -> int:
    return int(str(timestr).split(":")[0])

def compute_agents_markers(xlsx_path: str):
    df = pd.read_excel(Path(xlsx_path))
    df.columns = [c.strip() for c in df.columns]

    df = df[df["archetype"] == "PC_electric"].copy()
    df["hour"] = df["time_slot"].apply(parse_hour)
    df = df.sort_values(["agent", "hour"])

    home_in = df[df["todo"] == "Home_in"].copy()

    first_home_in = (
        home_in.groupby("agent", as_index=False)
        .first()[["agent", "hour", "mjkm"]]
        .rename(columns={"hour": "start_hour", "mjkm": "mj_consumed_at_marker"})
    )

    first_home_in["e_daily_kwh"] = first_home_in["mj_consumed_at_marker"] * MJ_TO_KWH

    return first_home_in[["agent", "start_hour", "e_daily_kwh"]]

# -------------------------------
# ASIGNAR BATERÍA Y SOC INICIAL
# -------------------------------
def assign_battery_and_initial_soc(
    agents_markers: pd.DataFrame,
    soc_min: float = 0.50,
    soc_max: float = 0.80,
    batt_mean_kwh: float = 60.0,
    batt_sd_kwh: float = 10.0,
    batt_min_kwh: float = 30.0,
    batt_max_kwh: float = 100.0,
    seed: int | None = 123
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = agents_markers.copy()

    batt = rng.normal(loc=batt_mean_kwh, scale=batt_sd_kwh, size=len(out))
    batt = np.clip(batt, batt_min_kwh, batt_max_kwh)
    out["battery_kwh"] = batt

    out["soc0"] = rng.uniform(low=soc_min, high=soc_max, size=len(out))
    return out

# -------------------------------
# SIMULACIÓN DE FLOTILLA CON REGLA SOC
# -------------------------------
def simulate_fleet_soc_rule(
    agents: pd.DataFrame,
    p_kw: float,
    n_days: int = 7,
    soc_threshold: float = 0.50,
    soc_target: float = 0.80
) -> np.ndarray:
    horizon = n_days * 24
    fleet_power = np.zeros(horizon, dtype=float)

    for _, row in agents.iterrows():
        start_hour = int(row["start_hour"])
        e_daily = float(row["e_daily_kwh"])
        batt = float(row["battery_kwh"])
        soc = float(row["soc0"])

        for d in range(n_days):
            t_arr = d * 24 + start_hour
            if t_arr >= horizon:
                break

            # Consumo diario
            soc -= e_daily / batt
            soc = max(soc, 0.0)

            # Si SOC < threshold, cargamos hasta target
            if soc < soc_threshold:
                e_need = max(0.0, (soc_target - soc) * batt)
                remaining = e_need
                for t in range(t_arr, horizon):
                    if remaining <= 0:
                        break
                    e_this = min(remaining, p_kw)
                    fleet_power[t] += e_this
                    remaining -= e_this

                soc = soc_target  # completamos idealmente

    return fleet_power

# -------------------------------
# PLOTEO TIPO "TUESDAY"
# -------------------------------
def plot_tuesday_curves(agents: pd.DataFrame, powers_kw: list[float], n_days: int = 7):
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    markers = ["o", "s", "^", "D", "*"]

    for i, p_kw in enumerate(powers_kw):
        fleet_48 = simulate_fleet_soc_rule(agents, p_kw=p_kw, n_days=2)  # 2 días
        tuesday = fleet_48[24:48]  # Día 2
        ax.plot(
            np.arange(24),
            tuesday,
            label=f"{p_kw:g} kW/EV",
            color=COLORS[i % len(COLORS)],
            marker=markers[i % len(markers)],
            linewidth=2.5,
            markersize=5,
        )

    ax.set_xlabel("Hour of day (Tuesday)")
    ax.set_ylabel("Aggregated charging power (kW)")
    ax.set_xticks(range(0, 24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend()
    plt.tight_layout()
    plt.show()

# -------------------------------
# MAIN
# -------------------------------
if __name__ == "__main__":
    xlsx = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\results\Kanaleneiland_schedule_vehicle_quantified_24.xlsx"
    agents_markers = compute_agents_markers(xlsx)
    agents = assign_battery_and_initial_soc(agents_markers)

    powers = [3.7, 7.4, 22.0, 50.0, 150.0]  # distintos escenarios de potencia

    plot_tuesday_curves(agents, powers, n_days=7)