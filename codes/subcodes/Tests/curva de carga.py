import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

MJ_TO_KWH = 1.0 / 3.6  # 0.277777...

# Convert size from mm to inches
fig_width = 368 / 25.4
fig_height = 78 / 25.4

def parse_hour(timestr: str) -> int:
    # expects "HH:MM"
    return int(str(timestr).split(":")[0])

def build_agent_profile(start_hour: int, e_need_kwh: float, p_max_kw: float, horizon_hours: int) -> np.ndarray:
    """
    Returns a power vector (kW) of length horizon_hours.
    Charging starts at start_hour with constant max power p_max_kw
    until recovering e_need_kwh. Last hour may be partial (average power in that hour).
    """
    prof = np.zeros(horizon_hours, dtype=float)

    if start_hour is None or (isinstance(e_need_kwh, float) and np.isnan(e_need_kwh)) or e_need_kwh <= 0:
        return prof
    if start_hour < 0 or start_hour >= horizon_hours:
        return prof

    remaining = float(e_need_kwh)
    for h in range(start_hour, horizon_hours):
        if remaining <= 0:
            break
        e_can = p_max_kw * 1.0  # kWh in 1 hour
        e_this = min(remaining, e_can)
        prof[h] = e_this / 1.0  # kW average over the hour
        remaining -= e_this

    return prof

def compute_agents_markers(
    xlsx_path: str,
    sheet_name: str | int | None = 0,
    day_filter: str | None = None
) -> pd.DataFrame:
    """
    Reads the schedule file and returns a per-agent table with:
      - agent
      - start_hour (first Home_in hour)
      - e_need_kwh (energy to recover at that marker)
    """
    xlsx_path = Path(xlsx_path)
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    df.columns = [c.strip() for c in df.columns]

    required_cols = {"time_slot", "agent", "archetype", "todo", "mjkm"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

    # only EVs
    df = df[df["archetype"] == "PC_electric"].copy()

    # optional day filter
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

    first_home_in["e_need_kwh"] = first_home_in["mj_consumed_at_marker"] * MJ_TO_KWH

    return first_home_in[["agent", "start_hour", "e_need_kwh"]]

def build_fleet_load_48h(agents_markers: pd.DataFrame, p_kw: float) -> pd.DataFrame:
    """
    Builds 48h aggregated fleet load for a single charging power p_kw.
    Replicates day-2 pattern by starting again at (24 + start_hour).
    """
    horizon = 48
    profiles = []

    for _, row in agents_markers.iterrows():
        start = int(row["start_hour"])
        e_need = float(row["e_need_kwh"])

        prof_d1 = build_agent_profile(start, e_need, p_kw, horizon)
        prof_d2 = build_agent_profile(24 + start, e_need, p_kw, horizon)
        profiles.append(prof_d1 + prof_d2)

    fleet = np.sum(np.vstack(profiles), axis=0) if profiles else np.zeros(horizon)

    out = pd.DataFrame({"t": np.arange(horizon), "P_kW": fleet})
    out["day"] = np.where(out["t"] < 24, "Mo", "Tu")
    out["hour"] = out["t"] % 24
    return out

def plot_tuesday_curves_for_powers(
    agents_markers: pd.DataFrame,
    powers_kw: list[float],
    title: str = "Aggregated charging load (Tuesday) with Monday spillover",
):

    # Convert size INSIDE the function (robusto)
    fig_width = 368 / 25.4
    fig_height = 78 / 25.4

    plt.figure(figsize=(fig_width/3.1, fig_height))

    for p in powers_kw:
        fleet_48 = build_fleet_load_48h(agents_markers, p_kw=p)
        tuesday = fleet_48[(fleet_48["t"] >= 24) & (fleet_48["t"] < 48)].copy()

        plt.plot(
            tuesday["hour"],
            tuesday["P_kW"],
            label=f"{p:g} kW/EV"
        )

    plt.xlabel("Time of day (Tuesday)")
    plt.ylabel("Aggregated power (kW)")

    plt.xticks(
        range(0, 24, 1),
        [f"{h:02d}:00" for h in range(24)],
        rotation=45
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    xlsx = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\results\Kanaleneiland_schedule_vehicle_quantified_24.xlsx"

    agents_markers = compute_agents_markers(xlsx_path=xlsx, day_filter=None)

    # Your requested charging power scenarios:
    powers = [3.7, 5.5, 7.5, 11.5, 50.0]
    #powers = [3.7, 50.0]

    plot_tuesday_curves_for_powers(
        agents_markers=agents_markers,
        powers_kw=powers,
    )

    import matplotlib.pyplot as plt

    # --- Choose fixed colors for each curve ---
    COLOR_3P7 = "#76a5af"
    COLOR_50  = "#0c343d"

    LINE_W = 2.5  # thickness

    # Build curves (48h) for each scenario
    fleet_48_3p7 = build_fleet_load_48h(agents_markers, p_kw=3.7)
    fleet_48_50  = build_fleet_load_48h(agents_markers, p_kw=50.0)

    # Keep only Tuesday: t in [24, 47]
    tue_3p7 = fleet_48_3p7[(fleet_48_3p7["t"] >= 24) & (fleet_48_3p7["t"] < 48)].copy()
    tue_50  = fleet_48_50[(fleet_48_50["t"] >= 24) & (fleet_48_50["t"] < 48)].copy()



    

    plt.figure(figsize=(fig_width, fig_height*2))

    plt.plot(
        tue_3p7["hour"], tue_3p7["P_kW"],
        label="3.7 kW/EV",
        color=COLOR_3P7,
        linewidth=LINE_W
    )

    plt.plot(
        tue_50["hour"], tue_50["P_kW"],
        label="50 kW/EV",
        color=COLOR_50,
        linewidth=LINE_W,
        linestyle="--"
    )

    plt.xlabel("Time of day")
    plt.ylabel("Aggregated power (kW)")

    plt.xticks(
        range(0, 24, 1),
        [f"{h:02d}:00" for h in range(24)],
        rotation=45
    )

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

