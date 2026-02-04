import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

MJ_TO_KWH = 1.0 / 3.6  # 0.277777...
P_SLOW_KW = 3.7
P_FAST_KW = 50.0


def parse_hour(timestr: str) -> int:
    # Espera formato "HH:MM"
    return int(str(timestr).split(":")[0])


def compute_agents_markers(
    xlsx_path: str,
    sheet_name: str | int | None = 0,
    day_filter: str | None = None,  # e.g. "Mo"
):
    """
    Lee el Excel, filtra VE (PC_electric), encuentra por agente su primer Home_in,
    y calcula consumo diario (kWh) hasta ese marcador.

    Devuelve:
      - agents_markers con columnas: agent, start_hour, e_daily_kwh
      - meta
    """
    xlsx_path = Path(xlsx_path)
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)

    df.columns = [c.strip() for c in df.columns]

    required_cols = {"time_slot", "agent", "archetype", "todo", "mjkm"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Faltan columnas requeridas: {missing}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    # Solo VE
    df = df[df["archetype"] == "PC_electric"].copy()

    # Filtro opcional por día
    if day_filter is not None and "day" in df.columns:
        df = df[df["day"] == day_filter].copy()

    df["hour"] = df["time_slot"].apply(parse_hour)
    df = df.sort_values(["agent", "hour"])

    home_in = df[df["todo"] == "Home_in"].copy()

    first_home_in = (
        home_in.groupby("agent", as_index=False)
        .first()[["agent", "hour", "mjkm"]]
        .rename(columns={"hour": "start_hour", "mjkm": "mj_consumed_at_marker"})
    )

    # Consumo diario hasta llegar a casa (kWh)
    first_home_in["e_daily_kwh"] = first_home_in["mj_consumed_at_marker"] * MJ_TO_KWH

    meta = {
        "n_agents_total_pc_electric": df["agent"].nunique(),
        "n_agents_with_home_in": len(first_home_in),
        "mj_to_kwh": MJ_TO_KWH,
    }

    return first_home_in[["agent", "start_hour", "e_daily_kwh"]], meta


def assign_battery_and_initial_soc(
    agents_markers: pd.DataFrame,
    soc_min: float = 0.50,
    soc_max: float = 0.80,
    batt_mean_kwh: float = 60.0,
    batt_sd_kwh: float = 10.0,
    batt_min_kwh: float = 30.0,
    batt_max_kwh: float = 100.0,
    seed: int | None = 123,
) -> pd.DataFrame:
    """
    Asigna a cada vehículo:
      - batería (kWh) ~ Normal(mean, sd) recortada a [min,max]
      - SoC inicial uniforme en [soc_min, soc_max]
    """
    rng = np.random.default_rng(seed)
    out = agents_markers.copy()

    batt = rng.normal(loc=batt_mean_kwh, scale=batt_sd_kwh, size=len(out))
    batt = np.clip(batt, batt_min_kwh, batt_max_kwh)
    out["battery_kwh"] = batt

    out["soc0"] = rng.uniform(low=soc_min, high=soc_max, size=len(out))

    return out


def simulate_fleet_soc_rule_multi_power(
    agents: pd.DataFrame,
    powers_kw: list[float],
    n_days: int = 7,
    soc_threshold: float = 0.50,
    soc_target: float = 0.80,
    day_names: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Same SoC rule as before, but simulated for an arbitrary list of charging powers.
    Each power is simulated independently (parallel what-if scenarios).
    """

    horizon = n_days * 24
    if day_names is None:
        day_names = [f"D{d+1}" for d in range(n_days)]

    # Initialize aggregated power profiles
    fleet_power = {
        p: np.zeros(horizon, dtype=float)
        for p in powers_kw
    }

    charge_events = {p: 0 for p in powers_kw}

    for _, row in agents.iterrows():
        start_hour = int(row["start_hour"])
        e_daily = float(row["e_daily_kwh"])
        batt = float(row["battery_kwh"])
        soc0 = float(row["soc0"])

        for p_kw in powers_kw:
            soc = soc0  # reset SoC per scenario

            for d in range(n_days):
                t_arr = d * 24 + start_hour
                if t_arr >= horizon:
                    break

                # Daily driving consumption
                soc -= e_daily / batt
                soc = max(soc, 0.0)

                if soc < soc_threshold:
                    e_need = max(0.0, (soc_target - soc) * batt)
                    if e_need > 0:
                        charge_events[p_kw] += 1
                        remaining = e_need

                        for t in range(t_arr, horizon):
                            if remaining <= 0:
                                break
                            e_this = min(remaining, p_kw)
                            fleet_power[p_kw][t] += e_this
                            remaining -= e_this

                        soc = soc_target  # idealized completion

    # Build output DataFrame
    out = pd.DataFrame({"t": np.arange(horizon)})
    for p_kw in powers_kw:
        out[f"P_{p_kw:.1f}_kW"] = fleet_power[p_kw]

    out["day_index"] = out["t"] // 24
    out["hour"] = out["t"] % 24
    out["day"] = out["day_index"].map(lambda i: day_names[int(i)])
    out["time_label"] = out["day"] + " " + out["hour"].map(lambda h: f"{int(h):02d}:00")

    meta = {
        "n_days": n_days,
        "soc_threshold": soc_threshold,
        "soc_target": soc_target,
        "powers_kw": powers_kw,
        "charge_events": charge_events,
        "n_agents": len(agents),
    }

    return out, meta


if __name__ == "__main__":
    xlsx = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\results\Kanaleneiland_schedule_vehicle_quantified_24.xlsx"

    # 1) Marcadores por agente (hora llegada + consumo diario)
    agents_markers, meta_read = compute_agents_markers(
        xlsx_path=xlsx,
        sheet_name=0,
        day_filter=None
    )
    print("META READ:", meta_read)
    print(agents_markers.head())

    # 2) Asigna batería y SoC inicial (Lunes 00:00)
    agents = assign_battery_and_initial_soc(
        agents_markers,
        soc_min=0.50,
        soc_max=0.80,
        batt_mean_kwh=60.0,   # promedio típico
        batt_sd_kwh=5.0,
        batt_min_kwh=30.0,
        batt_max_kwh=120.0,
        seed=453
    )
    print("\nAgents with battery and soc0:")
    print(agents.head())

    n_days = 70
    base_week = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
    day_names = (base_week * (n_days // 7)) + base_week[:(n_days % 7)]

    powers_kw = [3.7, 50]

    fleet, meta_sim = simulate_fleet_soc_rule_multi_power(
        agents=agents,
        powers_kw=powers_kw,
        n_days=n_days,
        soc_threshold=0.50,
        soc_target=0.80,
        day_names=day_names
    )

    plt.figure(figsize=(14, 5))

    # Plot all charging power scenarios
    for p_kw in powers_kw:
        plt.plot(
            fleet["t"],
            fleet[f"P_{p_kw:.1f}_kW"],
            label=f"{p_kw:.1f} kW charger"
        )

    # ---- X axis: hours of day every 2 hours ----
    t_max = fleet["t"].max()

    xticks = np.arange(0, t_max + 1, 2)
    xtick_labels = [f"{int(t % 24):02d}" for t in xticks]

    plt.xticks(xticks, xtick_labels)

    # ---- Vertical lines: day boundaries ----
    for d in range(0, int(t_max / 24) + 1):
        plt.axvline(x=d * 24, color="k", linewidth=0.5, alpha=0.3)

    # Labels and styling
    plt.xlabel("Hour of day")
    plt.ylabel("Aggregated charging power (kW)")
    plt.grid(True, axis="y")  # only horizontal grid
    plt.legend(ncol=2, fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.show()