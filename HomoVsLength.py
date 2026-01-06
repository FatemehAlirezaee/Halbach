""" 
Halbach magnetic field vs Length
 @author: Fatemeh Alirezaee 
 """

import os
import numpy as np
import magpylib as magpy
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from decimal import Decimal


Br = 1.45  # in T
s = 1/3  
mu_r = 1.05


param_sets = [
    {'ring_radius': Decimal('0.180'), 'N_magnets': 72, 'a': Decimal('0.010'), 'min_length_mm': 360, 'max_length_mm': 1440, 'label': 'r=0.180, N=72, a=0.010'},
    {'ring_radius': Decimal('0.180'), 'N_magnets': 48, 'a': Decimal('0.015'), 'min_length_mm': 360, 'max_length_mm': 1440, 'label': 'r=0.180, N=48, a=0.015'},
    {'ring_radius': Decimal('0.150'), 'N_magnets': 60, 'a': Decimal('0.010'), 'min_length_mm': 300, 'max_length_mm': 1200, 'label': 'r=0.150, N=60, a=0.010'}
]


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
        angle_deg = 2 * np.rad2deg(i * angle_step)  # Doubled for standard Halbach dipole
        mag.rotate_from_angax(angle=angle_deg, axis='x')
        pos = (0, float(radius) * np.cos(i * angle_step), float(radius) * np.sin(i * angle_step))
        mag.position = pos
        magnets.append(mag)
    return magpy.Collection(magnets)


summary_dfs = {}

for param in param_sets:
    ring_radius = param['ring_radius']
    N_magnets = param['N_magnets']
    a = param['a']
    min_length_mm = param['min_length_mm']
    max_length_mm = param['max_length_mm']
    label = param['label']


    a_mm = int(float(a) * 1000)
    step_mm = 20 if a_mm == 10 else 30

    magnetization = Br / (1 - s + s * mu_r)  # in T
    magnet_dim = (float(a), float(a), float(a))


    target_lengths_mm = list(range(min_length_mm, max_length_mm + 1, step_mm))


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
        configs.append((num_rings, gap_mm, gap))  # Include exact gap (Decimal)


    config_id = f"r{int(float(ring_radius)*1000)}_N{N_magnets}_a{int(float(a)*1000)}"
    config_dir = f"Config_{config_id}"
    plots_dir = f"Plots_{config_id}"
    field_dir = f"field_{config_id}"
    os.makedirs(config_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(field_dir, exist_ok=True)

    summary_data = []
    summary_csv = f"{field_dir}/Halbach_summary.csv"

 
    for num_rings, gap_mm, exact_gap in tqdm(configs, desc=f"Simulating Halbach stacks for {label}"):
        gap = exact_gap  # Use exact Decimal gap for calculations
        center_to_center = a + gap
        x_positions = np.array([float((Decimal(-(num_rings - 1)) / Decimal(2) + Decimal(i)) * center_to_center) for i in range(num_rings)])


        length = Decimal(num_rings) * a + Decimal(num_rings - 1) * gap
        length_mm = float(length * Decimal(1000))  # Exact due to Decimal


        rings = []
        for xpos in x_positions:
            ring = create_ring(ring_radius, N_magnets, magnet_dim, magnetization)
            ring.move((xpos, 0, 0))
            rings.append(ring)
        halbach = magpy.Collection(rings)


        # if not os.path.exists(html_file):
        #     for ring_c in halbach.sources:
        #         for mag in ring_c.sources:
        #             mag.style.model3d.showdefault = True
        #             mag.style.magnetization.show = True
        #             mag.style.magnetization.color = 'red'
        #     fig = magpy.show(halbach, backend='plotly', return_fig=True)
        #     fig.write_html(html_file)

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

        # axes = ['x', 'y', 'z']
        # components = ['Bx', 'By', 'Bz']
        # B_comps = [Bx, By, Bz]
        # for ax_idx, axis in enumerate(axes):
        #     if axis == 'x':
        #         mask = (np.abs(observers[:, 1]) < 1e-12) & (np.abs(observers[:, 2]) < 1e-12)
        #         sweep = observers[mask][:, 0]
        #     elif axis == 'y':
        #         mask = (np.abs(observers[:, 0]) < 1e-12) & (np.abs(observers[:, 2]) < 1e-12)
        #         sweep = observers[mask][:, 1]
        #     else:
        #         mask = (np.abs(observers[:, 0]) < 1e-12) & (np.abs(observers[:, 1]) < 1e-12)
        #         sweep = observers[mask][:, 2]
        #     for comp_idx, comp in enumerate(components):
        #         plot_file = f"{plots_dir}/Halbach_{num_rings}rings_length{length_mm}_{comp}_{axis}.png"
        #         if not os.path.exists(plot_file):
        #             plt.figure(figsize=(6, 4))
        #             comp_data = B_comps[comp_idx][mask] * 1000  # in mT
        #             if len(sweep) > 0:
        #                 sort_idx = np.argsort(sweep)
        #                 sweep = sweep[sort_idx]
        #                 comp_data = comp_data[sort_idx]
        #                 plt.ylim(np.min(comp_data), np.max(comp_data))
        #                 plt.plot(sweep * 1000, comp_data, '-o', label=f'{comp} [mT]')
        #             plt.xlabel(f'{axis}-axis [mm]')
        #             plt.ylabel(f'{comp} [mT]')
        #             plt.title(f'Halbach {num_rings} rings, length {length_mm} mm, {comp} along {axis}-axis')
        #             plt.grid(True)
        #             plt.legend()
        #             plt.tight_layout()
        #             plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        #             plt.close()

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(summary_csv, index=False, float_format='%.6e')


    summary_dfs[label] = summary_df


    for comp in ['By', 'Bx', 'Bz']:
        plot_file = f"{plots_dir}/H_{comp}_vs_length.png"
        if not os.path.exists(plot_file):
            plt.figure()
            plt.plot(summary_df['Length_mm'], summary_df[f'H_{comp}'], 'o-', label=f'H_{comp}')
            plt.xlabel('Length [mm]')
            plt.ylabel('Homogeneity [ppm]')
            plt.title(f'{comp} Homogeneity vs Length ({label})')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig("plot_file.png", dpi=600, bbox_inches='tight')
            plt.close()

    plot_file = f"{plots_dir}/H_B_vs_length.png"
    if not os.path.exists(plot_file):
        plt.figure()
        plt.plot(summary_df['Length_mm'], summary_df['H_B'], 'o-', label='H_B')
        plt.xlabel('Length [mm]')
        plt.ylabel('Homogeneity [ppm]')
        plt.title(f'B Magnitude Homogeneity vs Length ({label})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("plot_file.png, dpi=600", bbox_inches='tight')
        plt.close()

    abs_plot_file = f"{plots_dir}/Mean_Bx_Bz_vs_length.png"
    if not os.path.exists(abs_plot_file):
        plt.figure()
        plt.plot(summary_df['Length_mm'], summary_df['Mean_absBx'], 'o-', label='Mean(|Bx|)')
        plt.plot(summary_df['Length_mm'], summary_df['Mean_absBz'], 'o-', label='Mean(|Bz|)')
        plt.xlabel('Length [mm]')
        plt.ylabel('Field [T]')
        plt.title(f'Mean(|Bx|) & Mean(|Bz|) vs Length ({label})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("abs_plot_file.png", dpi=600, bbox_inches='tight')
        plt.close()

    by_plot_file = f"{plots_dir}/Mean_By_vs_length.png"
    if not os.path.exists(by_plot_file):
        plt.figure()
        plt.plot(summary_df['Length_mm'], summary_df['Mean_By'], 'o-', label='Mean(By)')
        plt.xlabel('Length [mm]')
        plt.ylabel('Field [T]')
        plt.title(f'Mean(By) vs Length ({label})')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("by_plot_file.png", dpi=600, bbox_inches='tight')
        plt.close()


annot_configs = {
    'r=0.180, N=48, a=0.015': [360.0, 720.0],
    'r=0.150, N=60, a=0.010': [300.0, 900.0],
    'r=0.180, N=72, a=0.010': [420.0, 600.0]
}


combined_h_by_file = "Plots_combined/H_By_vs_length.png"
os.makedirs("Plots_combined", exist_ok=True)
if not os.path.exists(combined_h_by_file):
    plt.figure()
    for label, df in summary_dfs.items():
        plt.plot(df['Length_mm'], df['H_By'], 'o-', label=label)
        if label in annot_configs:
            for sl in annot_configs[label]:
                row = df[df['Length_mm'] == sl]
                if not row.empty:
                    y = row['H_By'].values[0]
                    if not np.isnan(y):
                        plt.annotate(f'{y:.2f}', (sl, y), xytext=(10, 10), textcoords='offset points', fontsize=8, arrowprops=dict(arrowstyle='->'))
    plt.xlabel('Length [mm]')
    plt.ylabel('Homogeneity [ppm]')
    plt.title('H_By Homogeneity vs Length (All Configurations)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("combined_h_by_file.png", dpi=600, bbox_inches='tight')
    plt.close()


combined_mean_by_file = "Plots_combined/Mean_By_vs_length.png"
if not os.path.exists(combined_mean_by_file):
    plt.figure()
    for label, df in summary_dfs.items():
        plt.plot(df['Length_mm'], df['Mean_By'], 'o-', label=label)
        if label in annot_configs:
            for sl in annot_configs[label]:
                row = df[df['Length_mm'] == sl]
                if not row.empty:
                    y = row['Mean_By'].values[0]
                    if not np.isnan(y):
                        plt.annotate(f'{y:.6f}', (sl, y), xytext=(10, 10), textcoords='offset points', fontsize=8, arrowprops=dict(arrowstyle='->'))
    plt.xlabel('Length [mm]')
    plt.ylabel('Field [T]')
    plt.title('Mean(By) vs Length (All Configurations)')
    plt.grid(True)
    plt.legend(loc='upper right', bbox_to_anchor=(1, 0.95))
    plt.tight_layout()
    plt.savefig("combined_mean_by_file.png", dpi=600, bbox_inches='tight')
    plt.close()


combined_mean_absbx_file = "Plots_combined/Mean_absBx_vs_length.png"
if not os.path.exists(combined_mean_absbx_file):
    plt.figure()
    for label, df in summary_dfs.items():
        plt.plot(df['Length_mm'], df['Mean_absBx'], 'o-', label=label)
        if label in annot_configs:
            for sl in annot_configs[label]:
                row = df[df['Length_mm'] == sl]
                if not row.empty:
                    y = row['Mean_absBx'].values[0]
                    if not np.isnan(y):
                        plt.annotate(f'{y:.6f}', (sl, y), xytext=(10, 10), textcoords='offset points', fontsize=8, arrowprops=dict(arrowstyle='->'))
    plt.xlabel('Length [mm]')
    plt.ylabel('Field [T]')
    plt.title('Mean(|Bx|) vs Length (All Configurations)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("combined_mean_absbx_file.png", dpi=600, bbox_inches='tight')
    plt.close()


combined_mean_absbz_file = "Plots_combined/Mean_absBz_vs_length.png"
if not os.path.exists(combined_mean_absbz_file):
    plt.figure()
    for label, df in summary_dfs.items():
        plt.plot(df['Length_mm'], df['Mean_absBz'], 'o-', label=label)
        if label in annot_configs:
            for sl in annot_configs[label]:
                row = df[df['Length_mm'] == sl]
                if not row.empty:
                    y = row['Mean_absBz'].values[0]
                    if not np.isnan(y):
                        plt.annotate(f'{y:.6f}', (sl, y), xytext=(10, 10), textcoords='offset points', fontsize=8, arrowprops=dict(arrowstyle='->'))
    plt.xlabel('Length [mm]')
    plt.ylabel('Field [T]')
    plt.title('Mean(|Bz|) vs Length (All Configurations)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("combined_mean_absbz_file.png", dpi=600, bbox_inches='tight')

    plt.close()
