""" 
Halbach: magnetic field vs magnet aspect ratio
for diffeerent (r,N,a)
@author: Fatemeh Alirezaee
"""


import os
import numpy as np
import magpylib as magpy
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from decimal import Decimal
# ------------------------
# Parameters
# ------------------------
Br = 1.45 # in T
s = 1/3 # Cube Demagnetization
mu_r = 1.05
# Define the four configurations
param_sets = [
    {'ring_radius': Decimal('0.180'), 'N_magnets': 48, 'a': Decimal('0.015'), 'label': 'r=0.180, N=48, a=0.015'},
    {'ring_radius': Decimal('0.135'), 'N_magnets': 36, 'a': Decimal('0.015'), 'label': 'r=0.135, N=36, a=0.015'},
    {'ring_radius': Decimal('0.180'), 'N_magnets': 72, 'a': Decimal('0.010'), 'label': 'r=0.180, N=72, a=0.010'},
    {'ring_radius': Decimal('0.225'), 'N_magnets': 92, 'a': Decimal('0.010'), 'label': 'r=0.225, N=92, a=0.010'}
]
# DSV at (xd,yd,zd)
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
    angle_step = 2 * np.pi / N
    for i in range(N):
        mag = magpy.magnet.Cuboid(polarization=(0, magnetization, 0), dimension=magnet_dim)
        angle_deg = 2 * np.rad2deg(i * angle_step) # Doubled for standard Halbach dipole
        mag.rotate_from_angax(angle=angle_deg, axis='x')
        pos = (0, float(radius) * np.cos(i * angle_step), float(radius) * np.sin(i * angle_step))
        mag.position = pos
        magnets.append(mag)
    return magpy.Collection(magnets)
# Dictionary to hold summary dataframes for each config
summary_dfs = {}
for param in param_sets:
    ring_radius = param['ring_radius']
    N_magnets = param['N_magnets']
    a = param['a']
    label = param['label']
    # Set step_mm based on a
    a_mm = int(float(a) * 1000)
    step_mm = 20 if a_mm == 10 else 30
    magnetization = Br / (1 - s + s * mu_r) # in T
    magnet_dim = (float(a), float(a), float(a))
    # Calculate min and max lengths based on aspect ratios 1 to 3
    min_ar = Decimal('1')
    max_ar = Decimal('3')
    min_length = Decimal('2') * ring_radius * min_ar
    max_length = Decimal('2') * ring_radius * max_ar
    min_length_mm = int(float(min_length * Decimal('1000')))
    max_length_mm = int(float(max_length * Decimal('1000')))
    # Generate target lengths exactly
    target_lengths_mm = list(range(min_length_mm, max_length_mm + 1, step_mm))
    # Configurations
    configs = []
    for tlm in target_lengths_mm:
        tl = Decimal(tlm) / Decimal(1000)
        num_rings = int(tl // a)
        if num_rings < 2:
            continue
        spacer_count = num_rings - 1
        gap = (tl - Decimal(num_rings) * a) / Decimal(spacer_count)
        if gap < 0:
            continue
        gap_mm = round(float(gap * Decimal(1000)), 3)
        configs.append((num_rings, gap_mm, gap)) # Include exact gap (Decimal)
    # Dynamic directories based on parameters
    config_id = f"r{int(float(ring_radius)*1000)}_N{N_magnets}_a{int(float(a)*1000)}"
    config_dir = f"Config_{config_id}"
    plots_dir = f"Plots_{config_id}"
    field_dir = f"field_{config_id}"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(field_dir, exist_ok=True)
    summary_data = []
    summary_csv = f"{field_dir}/Halbach_summary.csv"
    # Loop over all configurations
    for num_rings, gap_mm, exact_gap in tqdm(configs, desc=f"Simulating Halbach stacks for {label}"):
        gap = exact_gap # Use exact Decimal gap for calculations
        center_to_center = a + gap
        x_positions = np.array([float((Decimal(-(num_rings - 1)) / Decimal(2) + Decimal(i)) * center_to_center) for i in range(num_rings)])
        # Calculate axial length exactly
        length = Decimal(num_rings) * a + Decimal(num_rings - 1) * gap
        length_mm = float(length * Decimal(1000)) # Exact due to Decimal
        aspect_ratio = float(length / (Decimal('2') * ring_radius))
        # Create stacked rings (lightweight, always do)
        rings = []
        for xpos in x_positions:
            ring = create_ring(ring_radius, N_magnets, magnet_dim, magnetization)
            ring.move((xpos, 0, 0))
            rings.append(ring)
        halbach = magpy.Collection(rings)
  
        csv_file = f"{field_dir}/Halbach_{num_rings}rings_length{length_mm}_DSV.csv"
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            Bx = df['Bx_T'].values
            By = df['By_T'].values
            Bz = df['Bz_T'].values
        else:
            B = np.array(halbach.getB(observers))
            Bx, By, Bz = B[:, 0], B[:, 1], B[:, 2]
            df = pd.DataFrame(np.hstack((observers * 1000, B)), columns=['x_mm', 'y_mm', 'z_mm', 'Bx_T', 'By_T', 'Bz_T'])
            df.to_csv(csv_file, index=False, float_format='%.6e')
        B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
        def H(B_comp):
            mean_val = np.mean(B_comp)
            if abs(mean_val) < 1e-12:
                return np.nan
            return (np.max(B_comp) - np.min(B_comp)) / mean_val * 1e6
        H_By_val = H(By)
        H_Bx_val = H(Bx)
        H_Bz_val = H(Bz)
        H_B_val = H(B_mag)
        Mean_By_val = np.mean(By)
        Mean_Bx_val = np.mean(Bx)
        Mean_Bz_val = np.mean(Bz)
        Mean_absBx_val = np.mean(np.abs(Bx))
        Mean_absBz_val = np.mean(np.abs(Bz))
        summary_data.append({
            'Rings': num_rings,
            'Gap_mm': gap_mm,
            'Length_mm': length_mm,
            'Aspect_Ratio': aspect_ratio,
            'H_By': H_By_val,
            'H_Bx': H_Bx_val,
            'H_Bz': H_Bz_val,
            'H_B': H_B_val,
            'Mean_By': Mean_By_val,
            'Mean_Bx': Mean_Bx_val,
            'Mean_Bz': Mean_Bz_val,
            'Mean_absBx': Mean_absBx_val,
            'Mean_absBz': Mean_absBz_val
        })
       
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_csv, index=False, float_format='%.6e')
    # Store the summary_df for combined plots
    summary_dfs[label] = summary_df
    # Generate individual summary plots for this config
    for comp in ['By', 'Bx', 'Bz']:
        plot_file = f"{plots_dir}/H_{comp}_vs_aspect_ratio.png"
        if not os.path.exists(plot_file):
            plt.figure()
            plt.plot(summary_df['Aspect_Ratio'], summary_df[f'H_{comp}'], 'o-', label=f'H_{comp}')
            plt.xlabel('Aspect Ratio')
            plt.ylabel('Homogeneity [ppm]')
            plt.title(f'{comp} Homogeneity vs Aspect Ratio ({label})')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig("plot_file.png" , dpi=600, bbox_inches='tight')
            plt.close()
    plot_file = f"{plots_dir}/H_B_vs_aspect_ratio.png"
    if not os.path.exists(plot_file):
        plt.figure()
        plt.plot(summary_df['Aspect_Ratio'], summary_df['H_B'], 'o-', label='H_B')
        plt.xlabel('Aspect Ratio')
        plt.ylabel('Homogeneity [ppm]')
        plt.title(f'B Magnitude Homogeneity vs Aspect Ratio ({label})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("plot_file.png", dpi=600, bbox_inches='tight')
        plt.close()
    abs_plot_file = f"{plots_dir}/Mean_Bx_Bz_vs_aspect_ratio.png"
    if not os.path.exists(abs_plot_file):
        plt.figure()
        plt.plot(summary_df['Aspect_Ratio'], summary_df['Mean_absBx'], 'o-', label='Mean(|Bx|)')
        plt.plot(summary_df['Aspect_Ratio'], summary_df['Mean_absBz'], 'o-', label='Mean(|Bz|)')
        plt.xlabel('Aspect Ratio')
        plt.ylabel('Field [T]')
        plt.title(f'Mean(|Bx|) & Mean(|Bz|) vs Aspect Ratio ({label})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("abs_plot_file.png", dpi=600, bbox_inches='tight')
        plt.close()
    by_plot_file = f"{plots_dir}/Mean_By_vs_aspect_ratio.png"
    if not os.path.exists(by_plot_file):
        plt.figure()
        plt.plot(summary_df['Aspect_Ratio'], summary_df['Mean_By'], 'o-', label='Mean(By)')
        plt.xlabel('Aspect Ratio')
        plt.ylabel('Field [T]')
        plt.title(f'Mean(By) vs Aspect Ratio ({label})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("by_plot_file.png", dpi=600, bbox_inches='tight')
        plt.close()
# Now generate combined plots for H_By, Mean_By, Mean_absBx, Mean_absBz
# Annotations specifications
annot_configs = {
    'r=0.135, N=36, a=0.015': [270.0, 540.0],
    'r=0.180, N=48, a=0.015': [360.0, 720.0],
    'r=0.180, N=72, a=0.010': [420.0, 600.0],
    'r=0.225, N=92, a=0.010': [500.0, 900.0]
}
# Combined H_By vs Aspect Ratio (using Length_mm for annotations, but plot vs Aspect_Ratio)
combined_h_by_file = "Plots_combined/H_By_vs_aspect_ratio.png"
os.makedirs("Plots_combined", exist_ok=True)
if not os.path.exists(combined_h_by_file):
    plt.figure()
    for label, df in summary_dfs.items():
        plt.plot(df['Aspect_Ratio'], df['H_By'], 'o-', label=label)
        if label in annot_configs:
            for sl in annot_configs[label]:
                row = df[df['Length_mm'] == sl]
                if not row.empty:
                    x_ar = row['Aspect_Ratio'].values[0]
                    y = row['H_By'].values[0]
                    if not np.isnan(y):
                        plt.annotate(f'{y:.2f}', (x_ar, y), xytext=(10, 10), textcoords='offset points', fontsize=8, arrowprops=dict(arrowstyle='->'))
    plt.xlabel('Aspect Ratio')
    plt.ylabel('Homogeneity [ppm]')
    plt.title('H_By Homogeneity vs Aspect Ratio (All Configurations)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("combined_h_by_file.png", dpi=600, bbox_inches='tight')
    plt.close()
# Combined Mean_By vs Aspect Ratio
combined_mean_by_file = "Plots_combined/Mean_By_vs_aspect_ratio.png"
if not os.path.exists(combined_mean_by_file):
    plt.figure()
    for label, df in summary_dfs.items():
        plt.plot(df['Aspect_Ratio'], df['Mean_By'], 'o-', label=label)
        if label in annot_configs:
            for sl in annot_configs[label]:
                row = df[df['Length_mm'] == sl]
                if not row.empty:
                    x_ar = row['Aspect_Ratio'].values[0]
                    y = row['Mean_By'].values[0]
                    if not np.isnan(y):
                        plt.annotate(f'{y:.6f}', (x_ar, y), xytext=(10, 10), textcoords='offset points', fontsize=8, arrowprops=dict(arrowstyle='->'))
    plt.xlabel('Aspect Ratio')
    plt.ylabel('Field [T]')
    plt.title('Mean(By) vs Aspect Ratio (All Configurations)')
    plt.grid(True)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=True, ncol=2)
    plt.tight_layout()
    plt.savefig("combined_mean_by_file.png", dpi=600, bbox_inches='tight')
    plt.close()
# Combined Mean_absBx vs Aspect Ratio
combined_mean_absbx_file = "Plots_combined/Mean_absBx_vs_aspect_ratio.png"
if not os.path.exists(combined_mean_absbx_file):
    plt.figure()
    for label, df in summary_dfs.items():
        plt.plot(df['Aspect_Ratio'], df['Mean_absBx'], 'o-', label=label)
        if label in annot_configs:
            for sl in annot_configs[label]:
                row = df[df['Length_mm'] == sl]
                if not row.empty:
                    x_ar = row['Aspect_Ratio'].values[0]
                    y = row['Mean_absBx'].values[0]
                    if not np.isnan(y):
                        plt.annotate(f'{y:.6f}', (x_ar, y), xytext=(10, 10), textcoords='offset points', fontsize=8, arrowprops=dict(arrowstyle='->'))
    plt.xlabel('Aspect Ratio')
    plt.ylabel('Field [T]')
    plt.title('Mean(|Bx|) vs Aspect Ratio (All Configurations)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("combined_mean_absbx_file.png", dpi=600, bbox_inches='tight')
    plt.close()
# Combined Mean_absBz vs Aspect Ratio
combined_mean_absbz_file = "Plots_combined/Mean_absBz_vs_aspect_ratio.png"
if not os.path.exists(combined_mean_absbz_file):
    plt.figure()
    for label, df in summary_dfs.items():
        plt.plot(df['Aspect_Ratio'], df['Mean_absBz'], 'o-', label=label)
        if label in annot_configs:
            for sl in annot_configs[label]:
                row = df[df['Length_mm'] == sl]
                if not row.empty:
                    x_ar = row['Aspect_Ratio'].values[0]
                    y = row['Mean_absBz'].values[0]
                    if not np.isnan(y):
                        plt.annotate(f'{y:.6f}', (x_ar, y), xytext=(10, 10), textcoords='offset points', fontsize=8, arrowprops=dict(arrowstyle='->'))
    plt.xlabel('Aspect Ratio')
    plt.ylabel('Field [T]')
    plt.title('Mean(|Bz|) vs Aspect Ratio (All Configurations)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("combined_mean_absbz_file.png", dpi=600, bbox_inches='tight')
    plt.close()
# Combined Mean_absBx / Mean_By vs Aspect Ratio
combined_ratio_file = "Plots_combined/Mean_absBx_over_Mean_By_vs_aspect_ratio.png"
if not os.path.exists(combined_ratio_file):
    plt.figure()
    for label, df in summary_dfs.items():
        ratio = df['Mean_absBx'] / df['Mean_By']
        plt.plot(df['Aspect_Ratio'], ratio, 'o-', label=label)
        if label in annot_configs:
            for sl in annot_configs[label]:
                row = df[df['Length_mm'] == sl]
                if not row.empty:
                    x_ar = row['Aspect_Ratio'].values[0]
                    y = row['Mean_absBx'].values[0] / row['Mean_By'].values[0]
                    if not np.isnan(y):
                        plt.annotate(f'{y:.6f}', (x_ar, y), xytext=(10, 10), textcoords='offset points', fontsize=8, arrowprops=dict(arrowstyle='->'))
    plt.xlabel('Aspect Ratio')
    plt.ylabel('Mean(|Bx|)/Mean(By)')
    plt.title('Mean(|Bx|)/Mean(By) vs Aspect Ratio (All Configurations)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("combined_ratio_file.png", dpi=600, bbox_inches='tight')
    plt.close()