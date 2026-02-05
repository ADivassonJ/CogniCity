import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution  # <--- IMPORTANTE

MJ_TO_KWH = 1.0 / 3.6  # 0.277777...

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

def simulate_fleet_soc_rule_multi_power_with_delay(
    agents: pd.DataFrame,
    powers_kw: list[float],
    n_days: int = 7,
    soc_threshold: float = 0.50,
    soc_target: float = 0.80,
    delay_mu_h: float = 2.0,
    delay_sigma_h: float = 1.0,
    seed_delay: int = 1234,
    apply_delay_to_last_power_only: bool = True,
) -> tuple[pd.DataFrame, dict]:

    rng = np.random.default_rng(seed_delay)
    horizon = n_days * 24

    fleet_power = {p: np.zeros(horizon) for p in powers_kw}
    peak_power = {}

    for _, row in agents.iterrows():
        start_hour = int(row["start_hour"])
        e_daily = float(row["e_daily_kwh"])
        batt = float(row["battery_kwh"])
        soc0 = float(row["soc0"])

        for idx, p_kw in enumerate(powers_kw):
            soc = soc0

            for d in range(n_days):
                t_arr = d * 24 + start_hour
                if t_arr >= horizon:
                    break

                # Consumo diario
                soc -= e_daily / batt
                soc = max(soc, 0.0)

                if soc < soc_threshold:
                    e_need = max(0.0, (soc_target - soc) * batt)
                    if e_need <= 0:
                        continue

                    # ---- RETRASO ALEATORIO ----
                    delay = 0
                    if (not apply_delay_to_last_power_only) or (idx == len(powers_kw) - 1):
                        delay = int(
                            max(0, rng.normal(delay_mu_h, delay_sigma_h))
                        )

                    t_start = t_arr + delay
                    if t_start >= horizon:
                        continue

                    remaining = e_need

                    for t in range(t_start, horizon):
                        if remaining <= 0:
                            break
                        fleet_power[p_kw][t] += p_kw
                        remaining -= p_kw

                    soc = soc_target

    out = pd.DataFrame({"t": np.arange(horizon)})
    for p_kw in powers_kw:
        out[f"P_{p_kw:.1f}_kW"] = fleet_power[p_kw]
        peak_power[p_kw] = fleet_power[p_kw].max()

    meta = {
        "n_days": n_days,
        "delay_mu_h": delay_mu_h,
        "delay_sigma_h": delay_sigma_h,
        "peak_power": peak_power,
    }

    return out, meta



WARMUP_DAYS = 2
WARMUP_HOURS = WARMUP_DAYS * 24


# -------------------------------------------------------------------------
# Mantenemos esta función IGUAL, pero asegúrate de que el 'seed' reinicie
# el generador dentro para que la función sea determinista para el optimizador
# (mismos inputs -> mismo output), aunque el optimizador sea estocástico.
# -------------------------------------------------------------------------
def simulate_single_power_with_delay(
    agents: pd.DataFrame,
    p_kw: float,
    n_days: int,
    soc_target: float,
    delay_mu_h: float,
    delay_sigma_h: float,
    soc_priority: float = 0.50,
    seed_delay: int = 1234,
):
    # Reiniciamos el RNG en cada llamada para que la comparación sea justa
    rng = np.random.default_rng(seed_delay)
    horizon = n_days * 24
    fleet_power = np.zeros(horizon)

    # --- NOTA DE RENDIMIENTO ---
    # iterrows es lento. Para optimización real, vectorizar esto sería ideal,
    # pero para cambiar solo el algoritmo de búsqueda, esto funciona.
    for _, row in agents.iterrows():
        start_hour = int(row["start_hour"])
        e_daily = float(row["e_daily_kwh"])
        batt = float(row["battery_kwh"])
        soc = float(row["soc0"])

        for d in range(n_days):
            t_arr = d * 24 + start_hour
            t_depart = (d + 1) * 24 + start_hour

            if t_arr >= horizon:
                break

            soc -= e_daily / batt
            soc = max(soc, 0.0)

            e_need = max(0.0, (soc_target - soc) * batt)
            if e_need <= 0:
                continue

            if soc < soc_priority:
                delay = 0
            else:
                # Nos aseguramos que delay no sea negativo
                raw_delay = rng.normal(delay_mu_h, delay_sigma_h)
                delay = int(max(0, raw_delay))

            t_start = t_arr + delay

            if t_start >= t_depart:
                continue

            remaining = e_need

            # Vectorización simple del bucle interno de tiempo para velocidad
            duration = int(np.ceil(remaining / p_kw))
            end_charge = min(t_start + duration, min(t_depart, horizon))
            
            # Carga "plana" simplificada para velocidad
            if end_charge > t_start:
                # Si es el último paso, ajustamos la energía exacta
                steps = end_charge - t_start
                energy_added = min(remaining, steps * p_kw)
                
                # Distribuir potencia (simplificación rápida)
                fleet_power[t_start:end_charge] += p_kw 
                
                # Ajuste fino del último slot si sobra potencia
                overcharge = (steps * p_kw) - energy_added
                if overcharge > 0 and end_charge < horizon:
                     fleet_power[end_charge-1] -= overcharge

                remaining -= energy_added

            if remaining <= 0:
                soc = soc_target

    return fleet_power

# -------------------------------------------------------------------------
# NUEVA FUNCIÓN DE OPTIMIZACIÓN ESTOCÁSTICA
# -------------------------------------------------------------------------
def optimize_delay_parameters_stochastic(
    agents,
    p_kw,
    n_days,
    soc_target,
    bounds_mu=(0, 24),     # Rango de búsqueda para mu
    bounds_sigma=(0, 12),  # Rango de búsqueda para sigma
    max_iter=20,           # Controla cuánto tiempo busca
    pop_size=10            # Tamaño de la población (agentes de búsqueda)
):
    print("Iniciando optimización estocástica (Differential Evolution)...")
    
    WARMUP_DAYS = 2
    WARMUP_HOURS = WARMUP_DAYS * 24

    # 1. Definimos la función objetivo (Cost function)
    # El optimizador intentará minimizar el valor que devuelve esta función.
    def objective_function(params):
        mu, sigma = params
        
        # Simulamos con los parámetros que prueba el algoritmo
        profile = simulate_single_power_with_delay(
            agents=agents,
            p_kw=p_kw,
            n_days=n_days,
            soc_target=soc_target,
            delay_mu_h=mu,
            delay_sigma_h=sigma,
            seed_delay=1234 # Importante: semilla fija para comparar peras con peras
        )
        
        # Nuestra métrica a minimizar: El PICO DE POTENCIA
        # (Ignoramos el warmup)
        peak = profile[WARMUP_HOURS:].max()
        return peak

    # 2. Ejecutamos Differential Evolution
    # bounds: lista de tuplas [(min_mu, max_mu), (min_sigma, max_sigma)]
    result = differential_evolution(
        objective_function, 
        bounds=[bounds_mu, bounds_sigma],
        maxiter=max_iter,    # Generaciones máximas
        popsize=pop_size,    # Multiplicador de población
        seed=412,        # Semilla del optimizador (para reproducibilidad)
        disp=True            # Muestra progreso en consola
    )

    # 3. Recuperamos el mejor perfil
    best_mu, best_sigma = result.x
    best_peak = result.fun
    
    print(f"Optimización terminada. Evaluaciones: {result.nfev}")

    # Re-simulamos una vez más para obtener el perfil completo para plotear
    best_profile = simulate_single_power_with_delay(
        agents=agents,
        p_kw=p_kw,
        n_days=n_days,
        soc_target=soc_target,
        delay_mu_h=best_mu,
        delay_sigma_h=best_sigma,
        seed_delay=1234
    )

    return {
        "mu": best_mu,
        "sigma": best_sigma,
        "peak": best_peak,
        "profile": best_profile
    }

if __name__ == "__main__":
    # ... (TÚ CÓDIGO DE CARGA DE DATOS IGUAL QUE ANTES) ...
    # Supongamos que ya tienes 'agents', 'n_days', 'powers_kw' cargados
    # ...
    
    xlsx = r"C:\Users\asier.divasson\Documents\GitHub\CogniCity\results\Kanaleneiland_schedule_vehicle_quantified_24.xlsx"
    agents_markers, _ = compute_agents_markers(xlsx_path=xlsx)
    agents = assign_battery_and_initial_soc(agents_markers)
    
    n_days = 70
    p_min = 3.7
    
    # --- EJECUTAR OPTIMIZACIÓN ESTOCÁSTICA ---
    best = optimize_delay_parameters_stochastic(
        agents=agents,
        p_kw=p_min,
        n_days=n_days,
        soc_target=0.80,
        bounds_mu=(0, 23),    # Busca mu entre 0h y 23h
        bounds_sigma=(0, 12), # Busca sigma entre 0h y 12h
        max_iter=100,          # Ajusta esto: más bajo = más rápido, menos preciso
        pop_size=5           # Ajusta esto: más bajo = menos exploración
    )

    print("=== OPTIMAL DELAY PARAMETERS (STOCHASTIC) ===")
    print(f"mu    = {best['mu']:.2f} h")
    print(f"sigma = {best['sigma']:.2f} h")
    print(f"peak  = {best['peak']:.1f} kW")

    # ... (CÓDIGO DE PLOTEO IGUAL) ...
    plt.figure(figsize=(14, 5))
    plt.plot(best["profile"], label=f"Optimized (Peak: {best['peak']:.1f} kW)")
    plt.legend()
    plt.show()