""" 
2D projection field map of orthogonal component
@author: Fatemeh Alirezaee 
""" 

import os
import numpy as np
import magpylib as magpy
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.interpolate import UnivariateSpline
from decimal import Decimal, getcontext

getcontext().prec = 10

# ---------- Constants ----------
Br = 1.45
s = 1/3
mu_r = 1.05
magnetization = Br / (1 - s + s * mu_r)

xd, yd, zd = 0.0, 0.0, 0.0
dsv_radius = 0.10
dsv_step = 0.010
x = np.arange(-dsv_radius, dsv_radius + dsv_step/2, dsv_step)
y = np.arange(-dsv_radius, dsv_radius + dsv_step/2, dsv_step)
z = np.arange(-dsv_radius, dsv_radius + dsv_step/2, dsv_step)
grid_points = np.array([(xi, yi, zi) for xi in x for yi in y for zi in z])
observers = grid_points + np.array([xd, yd, zd])
distances = np.linalg.norm(observers, axis=1)
observers = observers[distances <= dsv_radius]

param_sets = [
    #{'ring_radius': Decimal('0.180'), 'N_magnets': 48, 'a': Decimal('0.015'),'label': 'r=0.180, N=48, a=0.015'},
    #{'ring_radius': Decimal('0.135'), 'N_magnets': 36, 'a': Decimal('0.015'),'label': 'r=0.135, N=36, a=0.015'},
    {'ring_radius': Decimal('0.180'), 'N_magnets': 72, 'a': Decimal('0.010'),'label': 'r=0.180, N=72, a=0.010'},
    #{'ring_radius': Decimal('0.225'), 'N_magnets': 92, 'a': Decimal('0.010'),'label': 'r=0.225, N=92, a=0.010'}
]

target_ARs = [1.0, 1.2, 2.0]
gap_mm_list = [0, 5, 10, 20]
GENERATE_3D = True   # set to False to skip interactive HTMLs

def create_ring(radius, N, magnet_dim, magnetization):
    magnets = []
    angle_step = 2 * np.pi / N
    for i in range(N):
        mag = magpy.magnet.Cuboid(polarization=(0, magnetization, 0), dimension=magnet_dim)
        angle_deg = 2 * np.rad2deg(i * angle_step)
        mag.rotate_from_angax(angle=angle_deg, axis='x')
        pos = (0, float(radius) * np.cos(i * angle_step), float(radius) * np.sin(i * angle_step))
        mag.position = pos
        magnets.append(mag)
    return magpy.Collection(magnets)

#########################################
#Compute magnetic field components
for param in param_sets:
    ring_radius = param['ring_radius']
    N_magnets = param['N_magnets']
    a = param['a']
    label = param['label']

    config_id = f"r{int(float(ring_radius)*1000)}_N{N_magnets}_a{int(float(a)*1000)}"
    field_dir = os.path.join("field_data", config_id)
    os.makedirs(field_dir, exist_ok=True)

    a_dec = a
    a_float = float(a_dec)
    ring_radius_float = float(ring_radius)

    for target_AR in target_ARs:
        target_length = target_AR * 2 * ring_radius_float
        for gap_mm in gap_mm_list:
            gap = Decimal(gap_mm) / Decimal(1000)
            n_exact = (Decimal(target_length) + gap) / (a_dec + gap)
            n_rings = max(2, int(round(float(n_exact))))
            length_dec = n_rings * a_dec + (n_rings - 1) * gap
            length_float = float(length_dec)
            actual_AR = length_float / (2 * ring_radius_float)

            magnet_dim = (a_float, a_float, a_float)
            center_to_center = a_dec + gap
            x_positions = np.array([
                float((Decimal(-(n_rings - 1)) / 2 + Decimal(i)) * center_to_center)
                for i in range(n_rings)
            ])
            rings = []
            for xpos in x_positions:
                ring = create_ring(ring_radius, N_magnets, magnet_dim, magnetization)
                ring.move((xpos, 0, 0))
                rings.append(ring)
            halbach = magpy.Collection(rings)

            csv_name = (f"Halbach_AR{actual_AR:.3f}_gap{gap_mm}mm_"
                        f"rings{n_rings}_length{length_float*1000:.1f}mm.csv")
            csv_path = os.path.join(field_dir, csv_name)
            if not os.path.exists(csv_path):
                B = np.array(halbach.getB(observers))
                Bx, By, Bz = B[:, 0], B[:, 1], B[:, 2]
                df = pd.DataFrame(np.hstack((observers * 1000, B)),
                                  columns=['x_mm', 'y_mm', 'z_mm', 'Bx_T', 'By_T', 'Bz_T'])
                df.to_csv(csv_path, index=False, float_format='%.6e')

            # ============ 3D & slice interactive plots ============
            if GENERATE_3D:
                df = pd.read_csv(csv_path)
                base_name = os.path.splitext(csv_name)[0]

                # Full 3D maps (marker size 4 for continuity)
                html_3d_dir = os.path.join(field_dir, "3D_plots")
                os.makedirs(html_3d_dir, exist_ok=True)
                for comp, col in [('Bx', 'Bx_T'), ('By', 'By_T'), ('Bz', 'Bz_T')]:
                    vmin, vmax = df[col].min(), df[col].max()
                    fig = go.Figure(data=go.Scatter3d(
                        x=df['x_mm'], y=df['y_mm'], z=df['z_mm'],
                        mode='markers',
                        marker=dict(size=4, color=df[col],
                                    colorscale='Jet', cmin=vmin, cmax=vmax,
                                    colorbar=dict(title=f'{comp} (T)'))
                    ))
                    fig.update_layout(
                        title=f'{comp} – {label} AR={actual_AR:.3f} gap={gap_mm}mm',
                        scene=dict(xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)')
                    )
                    fig.write_html(os.path.join(html_3d_dir, f"{base_name}_{comp}_3D.html"))

                # Planar slices (marker size 5)
                html_slice_dir = os.path.join(field_dir, "3D_slices")
                os.makedirs(html_slice_dir, exist_ok=True)
                df_xy = df.loc[(df['z_mm'].abs() < dsv_step/2)]
                df_xz = df.loc[(df['y_mm'].abs() < dsv_step/2)]
                df_yz = df.loc[(df['x_mm'].abs() < dsv_step/2)]

                for comp, col in [('Bx', 'Bx_T'), ('By', 'By_T'), ('Bz', 'Bz_T')]:
                    vmin = df[col].min()
                    vmax = df[col].max()
                    for plane, plane_df, title_plane in [('XY', df_xy, 'Z≈0'), ('XZ', df_xz, 'Y≈0'), ('YZ', df_yz, 'X≈0')]:
                        if not plane_df.empty:
                            fig = go.Figure(data=go.Scatter3d(
                                x=plane_df['x_mm'], y=plane_df['y_mm'], z=plane_df['z_mm'],
                                mode='markers',
                                marker=dict(size=5, color=plane_df[col],
                                            colorscale='Jet', cmin=vmin, cmax=vmax,
                                            colorbar=dict(title=f'{comp} (T)'))
                            ))
                            fig.update_layout(
                                title=f'{comp} {plane}-plane ({title_plane}) – {base_name}',
                                scene=dict(xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)')
                            )
                            fig.write_html(os.path.join(html_slice_dir, f"{base_name}_{comp}_{plane}_slice.html"))

######################################
#2D scatter plots (volume projection) & axis line plots
plot_root = "scatter_plots"
axis_line_root = "axis_line_plots"
os.makedirs(plot_root, exist_ok=True)
os.makedirs(axis_line_root, exist_ok=True)

for param in param_sets:
    ring_radius = param['ring_radius']
    N_magnets = param['N_magnets']
    a = param['a']
    label = param['label']

    config_id = f"r{int(float(ring_radius)*1000)}_N{N_magnets}_a{int(float(a)*1000)}"
    field_dir = os.path.join("field_data", config_id)
    csv_files = [f for f in os.listdir(field_dir) if f.endswith('.csv')]
    if not csv_files:
        continue

    data_dict = {}
    for fname in csv_files:
        try:
            parts = fname[:-4].split('_')
            ar_str = parts[1]        # "AR1.000"
            gap_str = parts[2]       # "gap0mm"
            actual_AR = float(ar_str[2:])
            gap_mm = int(gap_str[3:-2])
        except:
            continue
        df = pd.read_csv(os.path.join(field_dir, fname))
        # Mean By for this configuration
        mean_By = df['By_T'].mean()
        # Ratios as percentages using mean(By)
        df['Bx_over_By_pct'] = (df['Bx_T'] / mean_By) * 100
        df['Bz_over_By_pct'] = (df['Bz_T'] / mean_By) * 100
        data_dict[(actual_AR, gap_mm)] = df

    # Individual scatter plots & axis line plots
    ind_dir = os.path.join(plot_root, config_id, "individual")
    axis_dir = os.path.join(axis_line_root, config_id)
    os.makedirs(ind_dir, exist_ok=True)
    os.makedirs(axis_dir, exist_ok=True)

    comps = [
        ('Bx_T', 'Bx (T)'),
        ('By_T', 'By (T)'),
        ('Bz_T', 'Bz (T)'),
        ('Bx_over_By_pct', 'Bx/mean(By) (%)'),
        ('Bz_over_By_pct', 'Bz/mean(By) (%)')
    ]
    coords = ['x_mm', 'y_mm', 'z_mm']
    coord_labels = ['x (mm)', 'y (mm)', 'z (mm)']

    for (actual_AR, gap_mm), df in data_dict.items():
        r_mm = int(float(ring_radius) * 1000)
        title_str = f"r={r_mm} mm, AR={actual_AR:.2f}, gap={gap_mm} mm"

        # ---- Scatter plots (volume projection) ----
        for col, clabel in comps:
            for coord, xylabel in zip(coords, coord_labels):
                fig, ax = plt.subplots()
                sc = ax.scatter(df[coord], df[col], c=df[col],
                                cmap='jet', s=1, alpha=0.8)
                cb = plt.colorbar(sc, ax=ax, label=clabel)
                ax.set_xlabel(xylabel)
                ax.set_ylabel(clabel)
                ax.set_title(title_str)
                plt.tight_layout()
                fname = f"{col}_vs_{coord}_AR{actual_AR:.3f}_gap{gap_mm}mm.png"
                plt.savefig(os.path.join(ind_dir, fname), dpi=600, bbox_inches='tight')
                plt.close(fig)

        #axis line plots
        tol = 0.5  # mm
        # X-axis (y≈0, z≈0)
        df_x_axis = df[(df['y_mm'].abs() < tol) & (df['z_mm'].abs() < tol)].sort_values('x_mm')
        # Y-axis (x≈0, z≈0)
        df_y_axis = df[(df['x_mm'].abs() < tol) & (df['z_mm'].abs() < tol)].sort_values('y_mm')
        # Z-axis (x≈0, y≈0)
        df_z_axis = df[(df['x_mm'].abs() < tol) & (df['y_mm'].abs() < tol)].sort_values('z_mm')

        for axis_name, axis_df, axis_coord, axis_label in [
            ('x', df_x_axis, 'x_mm', 'x (mm)'),
            ('y', df_y_axis, 'y_mm', 'y (mm)'),
            ('z', df_z_axis, 'z_mm', 'z (mm)')
        ]:
            if len(axis_df) >= 4:
                coord_vals = axis_df[axis_coord].values
                coord_smooth = np.linspace(coord_vals.min(), coord_vals.max(), 300)
                for col, clabel in comps:
                    y_vals = axis_df[col].values
                    spl = UnivariateSpline(coord_vals, y_vals, s=0, k=3)
                    y_smooth = spl(coord_smooth)
                    fig, ax = plt.subplots()
                    ax.plot(coord_smooth, y_smooth, '-', linewidth=2)
                    ax.set_xlabel(axis_label)
                    ax.set_ylabel(clabel)
                    ax.set_title(f"{title_str}  ({axis_name}-axis)")
                    ax.grid(True)
                    plt.tight_layout()
                    fname = f"smooth_{col}_vs_{axis_name}_AR{actual_AR:.3f}_gap{gap_mm}mm.png"
                    plt.savefig(os.path.join(axis_dir, fname), dpi=600, bbox_inches='tight')
                    plt.close(fig)
            elif len(axis_df) > 1:
                # fallback to simple line
                for col, clabel in comps:
                    fig, ax = plt.subplots()
                    ax.plot(axis_df[axis_coord], axis_df[col], '-', linewidth=2)
                    ax.set_xlabel(axis_label)
                    ax.set_ylabel(clabel)
                    ax.set_title(f"{title_str}  ({axis_name}-axis)")
                    ax.grid(True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(axis_dir, f"smooth_{col}_vs_{axis_name}_AR{actual_AR:.3f}_gap{gap_mm}mm.png"),
                                dpi=600, bbox_inches='tight')
                    plt.close(fig)

print("Halbach simulation complete.")