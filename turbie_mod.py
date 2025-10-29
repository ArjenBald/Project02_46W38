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