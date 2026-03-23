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

# Colores coherentes para todo el documento
COLORS = ["#76a5af", "#6aa84f", "#e69138", "#8e7cc3", "#0c343d"]

# -------------------------------
# FUNCIONES BASE
# -------------------------------

def parse_hour(timestr: str) -> int:
    return int(str(timestr).split(":")[0])


def build_agent_profile(start_hour: int, e_need_kwh: float, p_max_kw: float, horizon_hours: int) -> np.ndarray:

    prof = np.zeros(horizon_hours, dtype=float)

    if start_hour is None or np.isnan(e_need_kwh) or e_need_kwh <= 0:
        return prof
    if start_hour < 0 or start_hour >= horizon_hours:
        return prof

    remaining = float(e_need_kwh)

    for h in range(start_hour, horizon_hours):
        if remaining <= 0:
            break
        e_can = p_max_kw
        e_this = min(remaining, e_can)
        prof[h] = e_this
        remaining -= e_this

    return prof


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
        .rename(columns={"hour": "start_hour",
                         "mjkm": "mj_consumed_at_marker"})
    )

    first_home_in["e_need_kwh"] = first_home_in["mj_consumed_at_marker"] * MJ_TO_KWH

    return first_home_in[["agent", "start_hour", "e_need_kwh"]]


# -------------------------------
# BUILD FLEET LOAD N DAYS
# -------------------------------

def build_fleet_load_ndays(agents_markers, p_kw, n_days=70):

    horizon = 24 * n_days
    profiles = []

    for _, row in agents_markers.iterrows():
        start = int(row["start_hour"])
        e_need = float(row["e_need_kwh"])

        prof = np.zeros(horizon)

        for d in range(n_days):
            prof += build_agent_profile(start + 24*d, e_need, p_kw, horizon)

        profiles.append(prof)

    fleet = np.sum(np.vstack(profiles), axis=0)

    return fleet


# -------------------------------
# PREVIEW 70 DÍAS (MULTIGRÁFICO)
# -------------------------------

def preview_70_days(agents_markers, powers_kw, n_days=70):

    n = len(powers_kw)

    fig, axes = plt.subplots(
        nrows=n,
        ncols=1,
        figsize=(fig_width, fig_height * n),
        sharex=True
    )

    if n == 1:
        axes = [axes]

    for i, p in enumerate(powers_kw):

        fleet = build_fleet_load_ndays(agents_markers, p_kw=p, n_days=n_days)

        axes[i].plot(
            np.arange(len(fleet)),
            fleet,
            color=COLORS[i % len(COLORS)],
            linewidth=1.2
        )

        axes[i].set_ylabel(f"{p:g} kW")
        axes[i].grid(axis="y", linestyle="--", alpha=0.7)

        # estilo limpio coherente
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)

    axes[-1].set_xlabel("Time (hours)")

    plt.tight_layout()
    plt.show()


# -------------------------------
# FIGURA FINAL MARTES (TAMAÑO EXACTO)
# -------------------------------

def plot_tuesday_curves_for_powers(agents_markers, powers_kw):

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    markers = ["o", "s", "^", "D", "*"]

    for i, p in enumerate(powers_kw):

        fleet_48 = build_fleet_load_ndays(agents_markers, p_kw=p, n_days=2)

        tuesday = fleet_48[24:48]

        ax.plot(
            np.arange(24),
            tuesday,
            label=f"{p:g} kW/EV",
            color=COLORS[i % len(COLORS)],
            marker=markers[i % len(markers)],
            linewidth=2.5,
            markersize=5,
        )

    ax.set_xlabel("Time of day (Tuesday)")
    ax.set_ylabel("Aggregated power (kW)")

    ax.set_xticks(range(0, 24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45)

    ax.grid(axis='y', linestyle='--', alpha=0.7)

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

    powers = [3.7, 7.4, 22.0, 50.0, 150.0]  # mínimo arriba, máximo abajo

    # 🔹 1) PREVIEW 70 DÍAS
    preview_70_days(agents_markers, powers, n_days=70)

    # 🔹 2) FIGURA FINAL MARTES
    plot_tuesday_curves_for_powers(agents_markers, powers)