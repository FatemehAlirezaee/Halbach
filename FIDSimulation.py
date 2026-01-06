"""
FID simulation using Magnetic Field map
@author: Fatemeh Alirezaee
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Constants
gamma = 2 * np.pi * 42.58e6  # rad/s/T for 1H
t_max = 5e-4  # seconds
num_t = 10000
t = np.linspace(0, t_max, num_t)

# Function to compute FID and projections (vectorized for efficiency)
def compute_fid(Bx, By, Bz, include_projection=True):
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    B0 = np.mean(B_mag)
    delta_B = B_mag - B0
    if include_projection:
        proj = By / B_mag  # cos(theta)
    else:
        proj = np.ones_like(By)
    # Vectorized phase computation
    phases = np.exp(-1j * gamma * delta_B[:, np.newaxis] * t[np.newaxis, :])
    S = np.mean(proj[:, np.newaxis] * phases, axis=0)
    return np.abs(S), proj

# Load field maps (replace paths)
constrained_df = pd.read_csv('Constrained.csv')
unconstrained_df = pd.read_csv('Unconstrained.csv')

Bx_con, By_con, Bz_con = constrained_df['Bx_T'].values, constrained_df['By_T'].values, constrained_df['Bz_T'].values
Bx_unc, By_unc, Bz_unc = unconstrained_df['Bx_T'].values, unconstrained_df['By_T'].values, unconstrained_df['Bz_T'].values

# Compute FIDs and projs for constrained and unconstrained (full)
fid_con, proj_con = compute_fid(Bx_con, By_con, Bz_con)
fid_unc, proj_unc = compute_fid(Bx_unc, By_unc, Bz_unc)

# Simulate FID with only By (Bx=Bz=0)
fid_unc_by_only, _ = compute_fid(np.zeros_like(Bx_unc), By_unc, np.zeros_like(Bz_unc), include_projection=False)
fid_con_by_only, _ = compute_fid(np.zeros_like(Bx_con), By_con, np.zeros_like(Bz_con), include_projection=False)


# Initial amplitudes and losses (for full cases)
initial_con = fid_con[0]
initial_unc = fid_unc[0]
loss_con = 1 - np.mean(proj_con)
loss_unc = 1 - np.mean(proj_unc)
approx_loss_con = np.mean((Bx_con**2 + Bz_con**2) / (2 * By_con**2))
approx_loss_unc = np.mean((Bx_unc**2 + Bz_unc**2) / (2 * By_unc**2))

# Normalize FIDs to isolate dephasing decay
fid_con_norm = fid_con / fid_con[0]
fid_unc_norm = fid_unc / fid_unc[0]
fid_unc_by_only_norm = fid_unc_by_only / fid_unc_by_only[0]
fid_con_by_only_norm = fid_con_by_only / fid_con_by_only[0]

# Fit T2*: S_norm(t) ≈ exp(-t / T2*)
def exp_decay(t, T2star):
    return np.exp(-t / T2star)

# Safe fit function to handle non-decaying (homogeneous) cases
def safe_fit(t, fid_norm):
    if np.allclose(fid_norm, 1, atol=1e-10):  # No decay, perfect homogeneous
        return np.inf
    else:
        try:
            popt, _ = curve_fit(exp_decay, t, fid_norm, p0=[0.05],
                                ftol=1e-15, xtol=1e-15, gtol=1e-15, maxfev=10000)
            return popt[0] * 1000
        except:
            return np.nan

T2star_con_ms = safe_fit(t, fid_con_norm)
T2star_unc_ms = safe_fit(t, fid_unc_norm)
T2star_unc_by_only_ms = safe_fit(t, fid_unc_by_only_norm)
T2star_con_by_only_ms = safe_fit(t, fid_con_by_only_norm)


print(f"Constrained (full): T2* = {T2star_con_ms:.6f} ms" if np.isfinite(T2star_con_ms) else "Constrained (full): T2* = inf ms")
print(f"Unconstrained (full): T2* = {T2star_unc_ms:.6f} ms" if np.isfinite(T2star_unc_ms) else "Unconstrained (full): T2* = inf ms", f"reduced by {(1 - T2star_unc_ms / T2star_con_ms)*100:.1f}% compared to constrained" if np.isfinite(T2star_unc_ms) and np.isfinite(T2star_con_ms) else "")
print(f"Unconstrained (By only): T2* = {T2star_unc_by_only_ms:.6f} ms" if np.isfinite(T2star_unc_by_only_ms) else "Unconstrained (By only): T2* = inf ms")
print(f"Constrained (By only): T2* = {T2star_con_by_only_ms:.6f} ms" if np.isfinite(T2star_con_by_only_ms) else "Constrained (By only): T2* = inf ms")

# ---- Plot and save FID comparison with T2* annotation ----
plt.figure(figsize=(10, 6))
plt.plot(t * 1e3, fid_con, label=f'Constrained full (T2* = {T2star_con_ms:.6f} ms)' if np.isfinite(T2star_con_ms) else 'Constrained full (T2* = inf ms)')
plt.plot(t * 1e3, fid_unc, label=f'Unconstrained full (T2* = {T2star_unc_ms:.6f} ms)', linestyle='--' if np.isfinite(T2star_unc_ms) else 'Unconstrained full (T2* = inf ms)')
plt.plot(t * 1e3, fid_unc_by_only, label=f'Unconstrained By only (T2* = {T2star_unc_by_only_ms:.6f} ms)', linestyle=':' if np.isfinite(T2star_unc_by_only_ms) else 'Unconstrained By only (T2* = inf ms)')
plt.plot(t * 1e3, fid_con_by_only, label=f'Constrained By only (T2* = {T2star_con_by_only_ms:.6f} ms)', linestyle='-.' if np.isfinite(T2star_con_by_only_ms) else 'Constrained By only (T2* = inf ms)')

plt.xlabel('Time (ms)')
plt.ylabel('|FID|')
plt.title('FID Comparison with T2*')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('FID_comparison_T2star.png', dpi=600, bbox_inches='tight')
plt.close()