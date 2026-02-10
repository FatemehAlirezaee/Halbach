"""
Computing spherical harmonics coefficients
@author: Fatemeh Alirezaee
"""
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.cm import ScalarMappable
import matplotlib as mpl
from scipy.special import lpmv, factorial
from scipy.linalg import lstsq


mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.titlesize'] = 22
mpl.rcParams['axes.labelsize'] = 18
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['figure.titlesize'] = 28
def compute_real_sh(l, m, theta, phi, typ='c'):
    if m < 0:
        raise ValueError("m should be >=0 for real SH")
    sign = (-1)**m
    norm = np.sqrt((2*l + 1)/(4*np.pi) * factorial(l - m)/factorial(l + m))
    if m > 0:
        norm *= np.sqrt(2)
        norm *= sign
    P = lpmv(m, l, np.cos(theta))
    if m == 0:
        Y = norm * P
    elif typ == 'c':
        Y = norm * P * np.cos(m * phi)
    elif typ == 's':
        Y = norm * P * np.sin(m * phi)
    else:
        raise ValueError("typ must be 'c' or 's' for m>0")
    return Y
def compute_dP_dx(l, m, x):
    if m == 0 and l == 0:
        return 0
    sqrt_term = np.sqrt(1 - x**2)
    if sqrt_term == 0:
        return 0
    term1 = (l + m) * (l - m + 1) * lpmv(m-1, l, x)
    term2 = lpmv(m+1, l, x)
    dP_dx = (1/2) * (term1 - term2) / sqrt_term
    return dP_dx
def compute_basis_grad(l, m, typ, r, theta, phi, x, y, z):
    if r == 0:
        if l != 1:
            return np.zeros(3)
        theta = np.pi / 2
        if typ == 'c':
            phi = 0
        elif typ == 's':
            phi = np.pi / 2
        else:
            phi = 0
    Y = compute_real_sh(l, m, theta, phi, typ)
    if m == 0:
        trig_phi = 1
        dtrig_dphi = 0
    elif typ == 'c':
        trig_phi = np.cos(m * phi)
        dtrig_dphi = -m * np.sin(m * phi)
    else:
        trig_phi = np.sin(m * phi)
        dtrig_dphi = m * np.cos(m * phi)
    sign = (-1)**m if m > 0 else 1
    norm = np.sqrt((2*l + 1)/(4*np.pi) * factorial(l - m)/factorial(l + m))
    if m > 0:
        norm *= np.sqrt(2) * sign
    P = lpmv(m, l, np.cos(theta))
    dP_dx = compute_dP_dx(l, m, np.cos(theta))
    dP_dtheta = dP_dx * (-np.sin(theta))
    dY_dtheta = norm * dP_dtheta * trig_phi
    dY_dphi = norm * P * dtrig_dphi
    rl = r**l
    rlm1 = r**(l-1) if l > 0 else 0
    Br = - l * rlm1 * Y
    Btheta = - rlm1 * dY_dtheta
    Bphi = - rlm1 * (dY_dphi / np.sin(theta)) if np.sin(theta) != 0 else 0
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    Bx = sin_theta * cos_phi * Br + cos_theta * cos_phi * Btheta - sin_phi * Bphi
    By = sin_theta * sin_phi * Br + cos_theta * sin_phi * Btheta + cos_phi * Bphi
    Bz = cos_theta * Br - sin_theta * Btheta
    return np.array([Bx, By, Bz])
def compute_plateau_for_file(filepath, l_max_max=10, radius_mm=100, scale_to_m=True):
    df = pd.read_csv(filepath)
    points = df[['x_mm', 'y_mm', 'z_mm']].values
    B_meas_full = df[['Bx_T', 'By_T', 'Bz_T']].values
    if scale_to_m:
        points /= 1000
        radius = radius_mm / 1000
    else:
        radius = radius_mm
    r_full = np.sqrt(np.sum(points**2, axis=1))
    mask = r_full <= radius
    points_masked = points[mask]
    B_meas_masked = B_meas_full[mask]
    By_meas = B_meas_masked[:, 1]
    By0 = np.mean(By_meas)
    if By0 == 0:
        By0 = 1.0
    theta = np.arccos(points_masked[:,2] / r_full[mask])
    phi = np.arctan2(points_masked[:,1], points_masked[:,0])
    r = r_full[mask]
    n_points = len(points_masked)
    rel_rmses = []
    for l_max_cur in range(1, l_max_max + 1):
        # Build params up to current l_max
        params = []
        for l in range(l_max_cur + 1):
            params.append((l, 0, '0'))
            for m in range(1, l + 1):
                params.append((l, m, 'c'))
                params.append((l, m, 's'))
        n_params = len(params)
        M = np.zeros((3 * n_points, n_params))
        for k, (l, m, typ) in enumerate(params):
            for i in range(n_points):
                grad = compute_basis_grad(l, m, typ, r[i], theta[i], phi[i],
                                          points_masked[i,0], points_masked[i,1], points_masked[i,2])
                M[3*i : 3*i+3, k] = grad
        coeffs, _, _, _ = lstsq(M, B_meas_masked.flatten(), lapack_driver='gelsy')
        B_recon = (M @ coeffs).reshape(-1, 3)
        delta_By = B_recon[:,1] - By_meas
        rms_rel = np.sqrt(np.mean(delta_By**2)) / np.abs(By0)
        rel_rmses.append(rms_rel)
    return rel_rmses
def process_files(l_max=10, radius_mm=100, scale_to_m=True):
    aspect_ratio1 = 1
    aspect_ratio2 = 2
    aspect_ratio3 = 3
    aspect_ratios = [aspect_ratio1, aspect_ratio2, aspect_ratio3]
    filepath1 = r"C:\SH\aspect ratio=1.csv"
    filepath2 = r"C:\SH\aspect ratio=2.csv"
    filepath3 = r"C:\SH\aspect ratio=3.csv"
    filepaths = [filepath1, filepath2, filepath3]
    directory = os.path.dirname(filepath1)
    files = [os.path.basename(fp) for fp in filepaths]
    params = []
    for l in range(l_max + 1):
        params.append((l, 0, '0'))
        for m in range(1, l + 1):
            params.append((l, m, 'c'))
            params.append((l, m, 's'))
    n_params = len(params)
    coeffs_list = []
    uniform_idx = [k for k, p in enumerate(params) if p[0] == 1]
    ptp_ppms = []
    power_spectra = []
    for idx, filepath in enumerate(filepaths):
        df = pd.read_csv(filepath)
        points = df[['x_mm', 'y_mm', 'z_mm']].values
        B_meas_full = df[['Bx_T', 'By_T', 'Bz_T']].values
        if scale_to_m:
            points /= 1000
            radius = radius_mm / 1000
        else:
            radius = radius_mm
        r_full = np.sqrt(np.sum(points**2, axis=1))
        mask = r_full <= radius
        points_masked = points[mask]
        B_meas_masked = B_meas_full[mask]
        B_meas = B_meas_masked.flatten()
        n_points = len(points_masked)
        theta = np.arccos(points_masked[:,2] / r_full[mask])
        phi = np.arctan2(points_masked[:,1], points_masked[:,0])
        r = r_full[mask]
        M = np.zeros((3 * n_points, n_params))
        for k, (l, m, typ) in enumerate(params):
            for i in range(n_points):
                grad = compute_basis_grad(l, m, typ, r[i], theta[i], phi[i],
                                          points_masked[i,0], points_masked[i,1], points_masked[i,2])
                M[3*i : 3*i+3, k] = grad
        coeffs, _, _, _ = lstsq(M, B_meas, lapack_driver='gelsy')
        coeffs_list.append(coeffs)
        coeff_df = pd.DataFrame({'l': [p[0] for p in params],
                                 'm': [p[1] for p in params],
                                 'typ': [p[2] for p in params],
                                 'A': coeffs})
        coeff_df.to_csv(os.path.join(directory, f'sh_coeffs_{files[idx]}.csv'), index=False)
        B_recon = (M @ coeffs).reshape(-1, 3)
        coeffs_uniform = coeffs[uniform_idx]
        M_uniform = M[:, uniform_idx]
        B_uniform = (M_uniform @ coeffs_uniform).reshape(-1, 3)
        By_meas = B_meas_masked[:,1]
        By_uniform = B_uniform[:,1]
        delta_By = By_meas - By_uniform
        By0 = np.mean(By_meas)
        ptp_inhom = (np.max(By_meas) - np.min(By_meas)) / np.abs(By0) * 1e6 if By0 != 0 else 0
        print(f"H_By for {files[idx]}: {ptp_inhom:.2f} ppm")
        ptp_ppms.append(ptp_inhom)
        power = np.zeros(l_max + 1)
        for k, (ll, mm, ttyp) in enumerate(params):
            power[ll] += coeffs[k]**2
        power_spectra.append(power)
        
        
    #CONTOUR PLOTS
    planes = {
        'yz': {'fixed': 'x', 'coords': (1, 2), 'labels': ('y', 'z'),
               'phi_func': lambda yy: np.where(yy >= 0, np.pi/2, 3*np.pi/2)},
        'xz': {'fixed': 'y', 'coords': (0, 2), 'labels': ('x', 'z'),
               'phi_func': lambda xx: np.arctan2(0, xx)},
        'xy': {'fixed': 'z', 'coords': (0, 1), 'labels': ('x', 'y'),
               'phi_func': lambda yy, xx: np.arctan2(yy, xx)}
    }
    components = ['Bx', 'By', 'Bz']
    for plane_name, plane in planes.items():
        fig, axs = plt.subplots(3, 3, figsize=(23, 21), sharex=True, sharey=True,
                                gridspec_kw={'wspace': 0.01, 'hspace': 0.18})
        fig.suptitle(f'Deviation in {plane_name.upper()} Plane')
        grid1, grid2 = np.meshgrid(np.linspace(-radius, radius, 50),
                                   np.linspace(-radius, radius, 50))
        rr = np.sqrt(grid1**2 + grid2**2)
        mask_plane = rr <= radius
        if plane_name == 'yz':
            phi_plane = plane['phi_func'](grid1)
            points_plane = np.stack([np.zeros_like(grid1), grid1, grid2], axis=-1).reshape(-1, 3)
        elif plane_name == 'xz':
            phi_plane = plane['phi_func'](grid1)
            points_plane = np.stack([grid1, np.zeros_like(grid1), grid2], axis=-1).reshape(-1, 3)
        else:
            phi_plane = plane['phi_func'](grid2, grid1)
            points_plane = np.stack([grid1, grid2, np.zeros_like(grid1)], axis=-1).reshape(-1, 3)
        r_plane = rr.flatten()
        theta_plane = np.arccos(grid2 / rr).flatten()
        phi_plane = phi_plane.flatten()
        M_plane = np.zeros((3 * len(r_plane), n_params))
        for k, (l, m, typ) in enumerate(params):
            for i in range(len(r_plane)):
                if r_plane[i] == 0 and l != 1:
                    continue
                grad = compute_basis_grad(l, m, typ, r_plane[i], theta_plane[i], phi_plane[i],
                                          points_plane[i,0], points_plane[i,1], points_plane[i,2])
                M_plane[3*i:3*i+3, k] = grad
        delta_planes = []
        for coeffs in coeffs_list:
            B_recon_plane = (M_plane @ coeffs).reshape(-1, 3)
            coeffs_uniform = coeffs[uniform_idx]
            B_uniform_plane = (M_plane[:, uniform_idx] @ coeffs_uniform).reshape(-1, 3)
            delta_planes.append(B_recon_plane - B_uniform_plane)
        # Balanced SymLogNorm (max/100)
        norms = []
        for comp_idx in range(3):
            all_vals = np.concatenate([delta_planes[j][mask_plane.flatten(), comp_idx] for j in range(3)])
            max_abs = np.max(np.abs(all_vals))
            vmin = -max_abs
            vmax = max_abs
            if max_abs > 0:
                linthresh = 10 ** np.floor(np.log10(max_abs) - 2)
            else:
                linthresh = 1e-10
            norm = SymLogNorm(linthresh=linthresh, base=10, vmin=vmin, vmax=vmax)
            norms.append(norm)
        for map_idx in range(3):
            delta_plane = delta_planes[map_idx]
            for comp_idx in range(3):
                delta_grid = np.full_like(grid1, np.nan)
                delta_grid[mask_plane] = delta_plane[mask_plane.flatten(), comp_idx]
                axs[comp_idx, map_idx].contourf(
                    grid1, grid2, delta_grid,
                    norm=norms[comp_idx],
                    cmap='turbo'
                )
                title = f'{components[comp_idx]} Deviation - (aspect ratio {aspect_ratios[map_idx]})'
                axs[comp_idx, map_idx].set_title(title)
                axs[comp_idx, map_idx].set_xlabel(plane['labels'][0] + ' (m)')
                axs[comp_idx, map_idx].set_ylabel(plane['labels'][1] + ' (m)')
                axs[comp_idx, map_idx].set_aspect('equal')
        for comp_idx in range(3):
            sm = ScalarMappable(norm=norms[comp_idx], cmap='turbo')
            cbar = fig.colorbar(sm, ax=axs[comp_idx, :],
                                fraction=0.045, pad=0.005, shrink=1.0, aspect=25)
            cbar.set_label(f'Delta {components[comp_idx]} (T)')
        plt.tight_layout(rect=[0.01, 0.01, 0.92, 0.98])
        plt.savefig(os.path.join(directory, f'homogeneity_{plane_name}.png'), dpi=600, bbox_inches='tight')
        plt.close()
        
    #RMS vs L_max plot
    rel_rmses_list = []
    for filepath in filepaths:
        rel_rmses = compute_plateau_for_file(filepath, l_max_max=10, radius_mm=radius_mm, scale_to_m=scale_to_m)
        rel_rmses_list.append(rel_rmses)
    for idx, rel_rmses in enumerate(rel_rmses_list):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(range(1, 11), rel_rmses, marker='o')
        ax.set_xlabel('L_max', fontsize=14)
        ax.set_ylabel('Mean Relative RMSE', fontsize=14)
        ax.set_title(f'Reconstruction Error vs L_max for aspect ratio {aspect_ratios[idx]}', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(directory, f'plateau_error_Lmax_{aspect_ratios[idx]}.png'), dpi=600, bbox_inches='tight')
        plt.close()
    
    # Power spectrum plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for idx, power in enumerate(power_spectra):
        ax.plot(range(l_max + 1), np.log10(power + 1e-20), label=files[idx])
    ax.set_xlabel('Degree l')
    ax.set_ylabel('log10(Power)')
    ax.set_title('SH Power Spectrum')
    ax.legend()
    plt.savefig(os.path.join(directory, 'sh_power_spectrum.png'), dpi=600, bbox_inches='tight')
    plt.close()
if __name__ == "__main__":
    process_files()
