import re
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp

def load_parameters(filepath):
    """
    Loads turbine parameters from a custom text file format.
    """
    parameters = {}
    pat = re.compile(r"^\s*([0-9\.eE+-]+)\s*#\s*([A-Za-z0-9_]+)")
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            m = pat.match(line)
            if m:
                parameters[m.group(2)] = float(m.group(1))
    
    parameters.setdefault("rho", 1.22) # [kg/m3]
    
    if "Dr" in parameters:
        parameters["A"] = np.pi * (0.5 * parameters["Dr"]) ** 2
    
    # Check for essential parameters
    for k in ["mt", "mb", "k1", "k2", "c1", "c2", "A", "rho"]:
        if k not in parameters:
            raise ValueError(f"Missing parameter: {k}")
    return parameters

def load_ct_file(filepath):
    """
    Loads and sorts the CT vs. V data file.
    """
    df = pd.read_csv(filepath, sep=r"\s+", comment="#", header=None, names=["V", "CT"])
    return df.sort_values("V").reset_index(drop=True)

def load_wind_file(filepath):
    """
    Loads and sorts a wind time series file (Time, V).
    """
    # Use header=0 to correctly read the first line as column names
    df = pd.read_csv(filepath, sep=r"\s+", comment="#", header=0) 
    
    if df.shape[1] < 2:
        raise ValueError("Wind file must have at least 2 columns (time, V).")
    
    # Ensure columns are (Time, V) regardless of original names
    df = df.iloc[:, :2] 
    df.columns = ["Time", "V"]
    return df.sort_values("Time").reset_index(drop=True)

def build_system_matrices(p):
    """
    Builds the M, C, and K matrices as required by the task.
    """
    M = np.diag([p["mt"], p["mb"]])
    C = np.array([[ p["c2"],        -p["c2"]],
                  [-p["c2"], p["c1"]+p["c2"]]], dtype=float)
    K = np.array([[ p["k2"],        -p["k2"]],
                  [-p["k2"], p["k1"]+p["k2"]]], dtype=float)
    return M, C, K

# --- NEW FUNCTIONS FOR COMMIT 2 ---

def simulate_single_case(params, wind_df, ct_interp, t_skip=60.0):
    """
    Runs a single simulation case (one wind file).
    """
    # Per task requirement: Use a single CT based on the file's average wind speed
    v_avg = float(wind_df["V"].mean())
    ct_avg = float(ct_interp(v_avg))

    get_wind = interp1d(wind_df["Time"].values,
                        wind_df["V"].values,
                        kind="linear",
                        fill_value="extrapolate",
                        bounds_error=False)

    def rhs(t, y):
        # Defines the Equations of Motion for the solver
        xt, vt, xb, vb = y
        V = float(get_wind(t))
        # Force uses instantaneous V, but constant (average) CT
        Fa = 0.5 * params["rho"] * params["A"] * (V**2) * ct_avg
        xt_dd = (Fa - params["c2"]*(vt-vb) - params["k2"]*(xt-xb)) / params["mt"]
        xb_dd = ( params["c2"]*(vt-vb) + params["k2"]*(xt-xb) - params["c1"]*vb - params["k1"]*xb) / params["mb"]
        return [vt, xt_dd, vb, xb_dd]

    t_span = (float(wind_df["Time"].min()), float(wind_df["Time"].max()))
    t_eval = wind_df["Time"].values
    y0 = np.zeros(4)
    sol = solve_ivp(rhs, t_span, y0, t_eval=t_eval, rtol=1e-6, atol=1e-9)

    # Statistics after transient phase
    mask = sol.t >= t_skip
    stats = {
        "V_avg": v_avg,
        "CT_avg": ct_avg,
        "xt_mean": float(np.mean(sol.y[0, mask])),
        "xt_std":  float(np.std (sol.y[0, mask])),
        "xb_mean": float(np.mean(sol.y[2, mask])),
        "xb_std":  float(np.std (sol.y[2, mask])),
    }
    return sol, stats

def save_timeseries(path, sol):
    """
    Saves the full simulation time series to a text file.
    """
    df = pd.DataFrame({
        "Time(s)": sol.t,
        "xt(m)":   sol.y[0, :],
        "vt(m/s)": sol.y[1, :],
        "xb(m)":   sol.y[2, :],
        "vb(m/s)": sol.y[3, :],
    })
    df.to_csv(path, sep="\t", index=False, float_format="%.6f")