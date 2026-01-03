""" 
Halbach: magnetic field vs gap
for diffeerent magnet sizes, fixed d=36cm and L=42cm
@author: Fatemeh Alirezaee
"""


import os
import numpy as np
import magpylib as magpy
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================
# PARAMETER SETS (a, N)
# ============================================================
param_sets = [
    (0.010, 72),
    (0.012, 64),
    (0.015, 52),
    (0.020, 42)
]

ring_radius = 0.180
Br = 1.45
s = 1/3
mu_r = 1.05

# ============================================================
# STORAGE FOR COMPARISON PLOTS
# ============================================================
all_results = {}

# ============================================================
# LOOP OVER (a, N)
# ============================================================
for a, N_magnets in param_sets:

    magnetization = Br / (1 - s + s * mu_r)
    magnet_dim = (a, a, a)

    max_length = 0.42
    max_num_rings = int(max_length / a)
    configs = []
    for num_rings in range(max_num_rings, 1, -1):
        gap = (max_length - num_rings * a) / (num_rings - 1)
        if gap > a + a*0.5:
            break
        gap_mm = round(gap * 1000, 3)
        configs.append((num_rings, gap_mm))

    size_mm = int(a * 1000)
    os.makedirs(f"Config{size_mm}", exist_ok=True)
    os.makedirs(f"Plots{size_mm}", exist_ok=True)
    os.makedirs(f"field{size_mm}", exist_ok=True)

    # ---------------- DSV ----------------
    xd, yd, zd = 0.0, 0.0, 0.0
    dsv_radius = 0.10
    dsv_step = 0.010
    x = np.arange(-dsv_radius, dsv_radius + dsv_step / 2, dsv_step)
    y = np.arange(-dsv_radius, dsv_radius + dsv_step / 2, dsv_step)
    z = np.arange(-dsv_radius, dsv_radius + dsv_step / 2, dsv_step)
    grid_points = np.array([(xi, yi, zi) for xi in x for yi in y for zi in z])
    
    observers = grid_points + np.array([xd, yd, zd])
    
    distances = np.linalg.norm(observers, axis=1)
    observers = observers[distances <= dsv_radius]

    def create_ring(radius, N, magnet_dim, magnetization):
        magnets = []
        angle_step = 2*np.pi / N
        for i in range(N):
            mag = magpy.magnet.Cuboid(
                polarization=(0, magnetization, 0),
                dimension=magnet_dim
            )
            angle_deg = 2 * np.rad2deg(i*angle_step)
            mag.rotate_from_angax(angle=angle_deg, axis='x')
            mag.position = (
                0,
                radius*np.cos(i*angle_step),
                radius*np.sin(i*angle_step)
            )
            magnets.append(mag)
        return magpy.Collection(magnets)

    summary_data = []

    # ========================================================
    # LOOP OVER STACK CONFIGURATIONS
    # ========================================================
    for num_rings, gap_mm in tqdm(configs, desc=f"Simulating a={a}, N={N_magnets}"):

        gap = gap_mm / 1000
        center_to_center = a + gap
        x_positions = np.array([(-(num_rings-1)/2 + i)*center_to_center
                                for i in range(num_rings)])

        rings = []
        for xpos in x_positions:
            ring = create_ring(ring_radius, N_magnets,
                               magnet_dim, magnetization)
            ring.move((xpos, 0, 0))
            rings.append(ring)
        halbach = magpy.Collection(rings)

        # ---------- SAVE CONFIG HTML ----------
        html_file = f"Config{size_mm}/Halbach_{num_rings}rings_gap{gap_mm}.html"
        if not os.path.exists(html_file):
            fig = magpy.show(halbach, backend='plotly', return_fig=True)
            fig.write_html(html_file)

        # ---------- FIELD CSV ----------
        csv_file = f"field{size_mm}/Halbach_{num_rings}rings_gap{gap_mm}_DSV.csv"
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            Bx = df['Bx_T'].values
            By = df['By_T'].values
            Bz = df['Bz_T'].values
        else:
            B = np.array(halbach.getB(observers))
            Bx, By, Bz = B[:,0], B[:,1], B[:,2]
            df = pd.DataFrame(
                np.hstack((observers*1000, B)),
                columns=['x_mm','y_mm','z_mm','Bx_T','By_T','Bz_T']
            )
            df.to_csv(csv_file, index=False, float_format='%.6e')

        def H(Bc):
            m = np.mean(Bc)
            if abs(m) < 1e-12:
                return np.nan
            return (np.max(Bc) - np.min(Bc)) / m * 1e6

        summary_data.append({
            'Gap_mm': gap_mm,
            'Mean_By': np.mean(By),
            'Mean_absBx': np.mean(np.abs(Bx)),
            'Mean_absBz': np.mean(np.abs(Bz)),
            'H_By': H(By),
            'Ratio_Bx_By': np.mean(np.abs(Bx)) / np.mean(By) if np.mean(By) != 0 else np.nan
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(
        f"field{size_mm}/Halbach_summary.csv",
        index=False,
        float_format="%.6e"
    )

    all_results[(a, N_magnets)] = summary_df

# ============================================================
# Plots for all configurations
# ============================================================

# ---------- Mean(By) vs gap ----------
plt.figure()
for (a, N), df in all_results.items():
    plt.plot(df['Gap_mm'], df['Mean_By'], '-o',
             label=f'a={a*1000:.0f}mm, N={N}')
plt.xlabel('Gap [mm]')
plt.ylabel('Mean(By) [T]')
plt.title('Mean(By) vs Gap (All Configurations)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Mean_By_vs_gap_all_configs.png", dpi=600, bbox_inches='tight')
plt.close()

# ---------- H_By vs gap ----------
plt.figure()
for (a, N), df in all_results.items():
    plt.plot(df['Gap_mm'], df['H_By'], '-o',
             label=f'a={a*1000:.0f}mm, N={N}')
plt.xlabel('Gap [mm]')
plt.ylabel('H_By [ppm]')
plt.title('H_By vs Gap (All Configurations)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("H_By_vs_gap_all_configs.png", dpi=600, bbox_inches='tight')
plt.close()

# ---------- Mean(|Bx|) & Mean(|Bz|) vs gap ----------
plt.figure()
for (a, N), df in all_results.items():
    plt.plot(df['Gap_mm'], df['Mean_absBx'], '-o',
             label=f'|Bx| a={a*1000:.0f}mm, N={N}')
    
plt.xlabel('Gap [mm]')
plt.ylabel('Field [T]')
plt.title('Mean(|Bx|) vs Gap (All Configurations)')
plt.grid(True)
plt.legend(ncol=2)
plt.tight_layout()
plt.savefig("Mean_absBx_vs_gap_all_configs.png", dpi=600, bbox_inches='tight')
plt.close()

plt.figure()
for (a, N), df in all_results.items():
    plt.plot(df['Gap_mm'], df['Mean_absBz'], '-o',
             label=f'|Bx| a={a*1000:.0f}mm, N={N}')
    
plt.xlabel('Gap [mm]')
plt.ylabel('Field [T]')
plt.title('Mean(|Bz|) vs Gap (All Configurations)')
plt.grid(True)
plt.legend(ncol=2)
plt.tight_layout()
plt.savefig("Mean_absBz_vs_gap_all_configs.png", dpi=600, bbox_inches='tight')
plt.close()

plt.figure()
for (a, N), df in all_results.items():
    plt.plot(df['Gap_mm'], df['Ratio_Bx_By'], '-o',
             label=f'a={a*1000:.0f}mm, N={N}')
plt.xlabel('Gap [mm]')
plt.ylabel('Mean(abs(Bx)) / Mean(By)')
plt.title('Mean(abs(Bx)) / Mean(By) vs Gap (All Configurations)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Ratio_Bx_By_vs_gap_all_configs.png", dpi=600, bbox_inches='tight')
plt.close()
