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

# Build M, C, K matrices (as required by task)
M, C, K = tm.build_system_matrices(params)

# --- Run ONE example simulation (4 m/s, TI 0.1) ---

print("--- Testing Commit 2: Running single simulation for 4 m/s, TI 0.1 ---")

ti_label = "wind_TI_0.1"
v_avg_int = 4 # The wind speed we are simulating

# --- 1. Setup paths for this run ---
ti_out = outp / ti_label
ti_out.mkdir(exist_ok=True)
wind_file_path = inp / "wind_files" / ti_label / f"wind_{v_avg_int}_ms_TI_0.1.txt"

# --- 2. Load wind data ---
wind_df = tm.load_wind_file(wind_file_path)

# --- 3. Run simulation ---
sol, stats = tm.simulate_single_case(params, wind_df, ct_interp)

# --- 4. Save Timeseries Output ---
v_int = int(round(stats["V_avg"]))
timeseries_path = ti_out / f"timeseries_v{v_int}.txt"
tm.save_timeseries(timeseries_path, sol)
print(f"Saved timeseries to: {timeseries_path}")

# --- 5. Save Statistics Output ---
# (For a single run, this file will only have one row)
df_stats = pd.DataFrame([stats]) 
stats_path = ti_out / f"statistics_{ti_label}.txt"
df_stats.to_csv(stats_path, sep="\t", index=False, float_format="%.6f")
print(f"Saved statistics to: {stats_path}")

# --- 6. Plot one example ---
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
plot_path = ti_out / f"example_timeseries_v{v_int}_{ti_label}.png"
plt.savefig(plot_path, dpi=150)
plt.close(fig)
print(f"Saved plot to: {plot_path}")

print("--- Commit 2 Test Complete. 3 files saved. ---")