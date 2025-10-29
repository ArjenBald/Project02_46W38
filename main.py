from pathlib import Path
import turbie_mod as tm
from pprint import pprint # Using pprint for clean dictionary printing

# --- Setup Paths ---
base = Path(__file__).parent
inp  = base / "inputs"

print("--- Testing Commit 1: Loading Functions ---")

# --- 1. Test Parameter Loading ---
print("\n1. Loading parameters from 'turbie_parameters.txt'...")
params = tm.load_parameters(inp / "turbie_parameters.txt")
pprint(params)

# --- 2. Test CT File Loading ---
print("\n2. Loading CT file 'CT.txt'...")
ct_tab = tm.load_ct_file(inp / "CT.txt")
print(ct_tab.head())

# --- 3. Test Wind File Loading (one example) ---
print("\n3. Loading one wind file ('wind_4_ms_TI_0.1.txt')...")
wind_file_path = inp / "wind_files" / "wind_TI_0.1" / "wind_4_ms_TI_0.1.txt"
wind_df = tm.load_wind_file(wind_file_path)
print(wind_df.head())

# --- 4. Test Matrix Building ---
print("\n4. Building system matrices...")
M, C, K = tm.build_system_matrices(params)
print(f"  M matrix:\n{M}")
print(f"  C matrix:\n{C}")
print(f"  K matrix:\n{K}")

print("\n--- Commit 1 Test Complete. All modules loaded. ---")