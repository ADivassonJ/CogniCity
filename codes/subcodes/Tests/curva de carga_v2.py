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

    out["soc0"] = rng.uniform(low=soc_min, high=soc_max, size=len(out))  # fracción 0..1

    return out


def simulate_fleet_soc_rule(
    agents: pd.DataFrame,
    n_days: int = 7,
    soc_threshold: float = 0.50,   # si SoC < 50% -> carga
    soc_target: float = 0.80,      # si carga, carga hasta 80%
    p_slow_kw: float = P_SLOW_KW,
    p_fast_kw: float = P_FAST_KW,
    day_names: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Regla:
      - Lunes 00:00 cada vehículo empieza con SoC aleatorio en [50%,80%] (ya asignado en agents['soc0'])
      - Cada día, al llegar a casa (start_hour), se resta el consumo del día al SoC
      - Si SoC < 50% -> empieza a cargar desde esa hora hasta alcanzar 80% (o hasta fin de horizonte)
      - Si SoC >= 50% -> no carga ese día

    La carga se modela hora a hora y puede hacer spillover al día siguiente (igual que antes).
    """
    horizon = n_days * 24
    if day_names is None:
        day_names = [f"D{d+1}" for d in range(n_days)]
    else:
        if len(day_names) != n_days:
            raise ValueError(f"day_names debe tener longitud n_days={n_days}")

    fleet_slow = np.zeros(horizon, dtype=float)
    fleet_fast = np.zeros(horizon, dtype=float)

    # Métricas
    n_charge_events_slow = 0
    n_charge_events_fast = 0

    for _, row in agents.iterrows():
        start_hour = int(row["start_hour"])
        e_daily = float(row["e_daily_kwh"])
        batt = float(row["battery_kwh"])
        soc = float(row["soc0"])  # fracción 0..1

        # Simula día a día (decisión en llegada a casa)
        for d in range(n_days):
            t_arr = d * 24 + start_hour
            if t_arr >= horizon:
                break

            # Al llegar a casa: baja SoC por consumo del día
            soc -= (e_daily / batt)
            soc = max(soc, 0.0)

            # Decide si carga
            if soc < soc_threshold:
                # Energía necesaria para llegar a soc_target
                e_need = max(0.0, (soc_target - soc) * batt)

                # --- SLOW charging ---
                if e_need > 0:
                    n_charge_events_slow += 1
                    remaining = e_need
                    for t in range(t_arr, horizon):
                        if remaining <= 0:
                            break
                        e_can = p_slow_kw * 1.0
                        e_this = min(remaining, e_can)
                        fleet_slow[t] += e_this / 1.0  # kW promedio
                        remaining -= e_this
                    # Nota: no actualizamos soc aquí para slow, porque el SoC real dependería del cargador elegido.
                    # El SoC para la decisión diaria lo simularemos con FAST o SLOW? Para mantener coherencia,
                    # simulamos el SoC con una "política" única: usar FAST para actualizar SoC sería inconsistente.
                    # Solución: actualizamos SoC con una carga "ideal" hasta target (sin importar potencia),
                    # ya que la decisión diaria depende del estado al día siguiente.
                    # Pero para spillover, el perfil ya está en fleet_slow.
                    soc = min(soc_target, soc + e_need / batt)

                # --- FAST charging ---
                if e_need > 0:
                    n_charge_events_fast += 1
                    remaining = e_need
                    for t in range(t_arr, horizon):
                        if remaining <= 0:
                            break
                        e_can = p_fast_kw * 1.0
                        e_this = min(remaining, e_can)
                        fleet_fast[t] += e_this / 1.0  # kW promedio
                        remaining -= e_this

                # Importante: el SoC que se usa para el día siguiente ya lo dejamos en soc_target.
                # Si quieres que el SoC dependa de si con SLOW llega o no a tiempo, hay que modelar
                # ventanas de carga (p.ej., hasta la hora de salida). Como no tenemos salida, usamos este supuesto.

            # Si soc >= threshold, no carga ese día

    out = pd.DataFrame({
        "t": np.arange(horizon),
        "P_slow_kW": fleet_slow,
        "P_fast_kW": fleet_fast,
    })
    out["day_index"] = out["t"] // 24
    out["hour"] = out["t"] % 24
    out["day"] = out["day_index"].map(lambda i: day_names[int(i)])
    out["time_label"] = out["day"] + " " + out["hour"].map(lambda h: f"{int(h):02d}:00")

    meta = {
        "n_days": n_days,
        "horizon_hours": horizon,
        "soc_threshold": soc_threshold,
        "soc_target": soc_target,
        "p_slow_kw": p_slow_kw,
        "p_fast_kw": p_fast_kw,
        "n_agents": int(agents["agent"].nunique()) if "agent" in agents.columns else len(agents),
        "n_charge_events_slow": n_charge_events_slow,
        "n_charge_events_fast": n_charge_events_fast,
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
        batt_sd_kwh=10.0,
        batt_min_kwh=30.0,
        batt_max_kwh=100.0,
        seed=123
    )
    print("\nAgents with battery and soc0:")
    print(agents.head())

    n_days = 70
    base_week = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
    day_names = (base_week * (n_days // 7)) + base_week[:(n_days % 7)]

    fleet_7d, meta_sim = simulate_fleet_soc_rule(
        agents=agents,
        n_days=n_days,
        soc_threshold=0.50,
        soc_target=0.80,
        p_slow_kw=P_SLOW_KW,
        p_fast_kw=P_FAST_KW,
        day_names=day_names
    )
    
    print("\nMETA SIM:", meta_sim)
    print("\nFleet 7d head:")
    print(fleet_7d.head(30))
    print("\nFleet 7d tail:")
    print(fleet_7d.tail(30))

    # 4) Plot 168h (muestra todos los días)
    plt.figure()
    plt.plot(fleet_7d["t"], fleet_7d["P_slow_kW"], label="Lenta (3.7 kW/VE)")
    plt.plot(fleet_7d["t"], fleet_7d["P_fast_kW"], label="Rápida (50 kW/VE)")

    ticks = np.arange(0, 7 * 24, 24)
    ticklabels = [fleet_7d.loc[fleet_7d["t"] == t, "day"].iloc[0] for t in ticks]
    plt.xticks(ticks, ticklabels)

    plt.xlabel("Día")
    plt.ylabel("Potencia agregada (kW)")
    plt.title("Curva agregada 7 días con regla SoC (<50% carga hasta 80%)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 5) (Opcional) Curvas slow por día (0..23)
    plt.figure()
    for d in range(7):
        day_df = fleet_7d[fleet_7d["day_index"] == d]
        plt.plot(day_df["hour"], day_df["P_slow_kW"], label=f"{day_df['day'].iloc[0]} (slow)")

    plt.xlabel("Hora del día")
    plt.ylabel("Potencia agregada (kW)")
    plt.title("Curvas slow por día (7 días)")
    plt.xticks(range(0, 24, 1))
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()

    # 5) Curvas solapadas: solo 1 de cada 6 días (día 6, 12, 18, ...)
    days_to_plot = list(range(5, n_days, 6))  # 0-based: 5->día 6, 11->día 12, ...

    # --- SLOW solapado ---
    plt.figure()
    for d in days_to_plot:
        day_df = fleet_7d[fleet_7d["day_index"] == d]
        if day_df.empty:
            continue
        plt.plot(day_df["hour"], day_df["P_slow_kW"], label=f"{day_df['day'].iloc[0]} (día {d+1})")

    plt.xlabel("Hora del día")
    plt.ylabel("Potencia agregada (kW)")
    plt.title("Curvas SLOW solapadas: días 6, 12, 18, ...")
    plt.xticks(range(0, 24, 1))
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()

    # --- FAST solapado ---
    plt.figure()
    for d in days_to_plot:
        day_df = fleet_7d[fleet_7d["day_index"] == d]
        if day_df.empty:
            continue
        plt.plot(day_df["hour"], day_df["P_fast_kW"], label=f"{day_df['day'].iloc[0]} (día {d+1})")

    plt.xlabel("Hora del día")
    plt.ylabel("Potencia agregada (kW)")
    plt.title("Curvas FAST solapadas: días 6, 12, 18, ...")
    plt.xticks(range(0, 24, 1))
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()
