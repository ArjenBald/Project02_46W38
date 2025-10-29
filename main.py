from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d as _interp1d # Import the interpolator directly

import turbie_mod as tm

# --- Setup Paths ---
base = Path(__file__).parent
inp  = base / "inputs"
outp = base / "outputs"
outp.mkdir(parents=True, exist_ok=True)

# --- Load Shared Data ---
params = tm.load_parameters(inp / "turbie_parameters.txt")
ct_tab = tm.load_ct_file(inp / "CT.txt")
ct_interp = _interp1d(ct_tab["V"].values, ct_tab["CT"].values, kind="linear", fill_value="extrapolate", bounds_error=False)

# Build M, C, K matrices as required by task (not used in RHS)
M, C, K = tm.build_system_matrices(params)

example_done = False

# --- Main Simulation Loop ---
for ti_dir in sorted((inp / "wind_files").iterdir()):
    if not ti_dir.is_dir():
        continue
    
    ti_label = ti_dir.name      # e.g., 'wind_TI_0.1'
    ti_out = outp / ti_label
    ti_out.mkdir(exist_ok=True)

    stats_rows = []

    # Sort wind files by average velocity number in the filename
    def _vnum(p):
        m = re.search(r"wind_([0-9]+)_ms", p.name)
        return int(m.group(1)) if m else 0

    for wf in sorted(ti_dir.glob("*.txt"), key=_vnum):
        
        # --- Run Simulation ---
        wind_df = tm.load_wind_file(wf)
        sol, stats = tm.simulate_single_case(params, wind_df, ct_interp)

        # --- Save Timeseries Output ---
        v_int = int(round(stats["V_avg"]))
        tm.save_timeseries(ti_out / f"timeseries_v{v_int}.txt", sol)
        stats_rows.append(stats)

        # --- Plot one example (first wind file) for displacements and wind ---
        if not example_done:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            axw = ax1.twinx()
            ax1.plot(sol.t, sol.y[0], label="xt")
            ax1.plot(sol.t, sol.y[2], label="xb")
            axw.plot(wind_df["Time"], wind_df["V"], ls=":", label="V(t)")
            ax1.set_title(f"Displacements & Wind (v≈{stats['V_avg']:.1f} m/s, {ti_label})")
            ax1.set_ylabel("Disp [m]"); axw.set_ylabel("Wind [m/s]")
            ax1.grid(True); ax1.legend(loc="upper left"); axw.legend(loc="upper right")
            
            ax2.plot(sol.t, sol.y[0], label="xt"); ax2.plot(sol.t, sol.y[2], label="xb")
            ax2.set_xlabel("Time [s]"); ax2.set_ylabel("Disp [m]"); ax2.grid(True); ax2.legend()
            
            plt.tight_layout()
            plt.savefig(ti_out / f"example_timeseries_v{v_int}_{ti_label}.png", dpi=150)
            plt.close(fig)
            example_done = True

    # --- Save summary statistics for this TI category ---
    df = pd.DataFrame(stats_rows).sort_values("V_avg").reset_index(drop=True)
    df.to_csv(ti_out / f"statistics_{ti_label}.txt", sep="\t", index=False, float_format="%.6f")

    # --- Plot Mean & Std vs. V_avg ---
    if not df.empty:
        # Plot Means vs V_avg
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(df["V_avg"], df["xt_mean"], marker="o", label="xt mean")
        ax1.plot(df["V_avg"], df["xb_mean"], marker="o", label="xb mean")
        ax1.set_xlabel("Avg wind [m/s]"); ax1.set_ylabel("Mean disp [m]")
        ax1.set_title(f"Means vs Wind Speed ({ti_label})"); ax1.grid(True); ax1.legend()
        plt.tight_layout(); plt.savefig(ti_out / f"means_{ti_label}.png", dpi=150); plt.close(fig1)

        # Plot Stds vs V_avg
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.plot(df["V_avg"], df["xt_std"], marker="o", label="xt std")
        ax2.plot(df["V_avg"], df["xb_std"], marker="o", label="xb std")
        ax2.set_xlabel("Avg wind [m/s]"); ax2.set_ylabel("Std disp [m]")
        ax2.set_title(f"Stds vs Wind Speed ({ti_label})"); ax2.grid(True); ax2.legend()
        plt.tight_layout(); plt.savefig(ti_out / f"stds_{ti_label}.png", dpi=150); plt.close(fig2)