"""
Genetic Algorithm (GA) optimization of a Halbach array dedicated for human's brain MRI
@author: Fatemeh Alirezaee
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import random
import time
from functools import lru_cache
import matplotlib.pyplot as plt
import gc
import pickle
# =====================================================
# SETTINGS
# =====================================================
BASE_DIR = r"C:\Design\Network_Data"
CONFIG_DIR = os.path.join(BASE_DIR, "Configurations")
OUTPUT_DIR = os.path.join(BASE_DIR, "GA_Output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
BLOCK_IDS = list(range(1, 71))
MIN_BLOCKS = 17
MAX_BLOCKS = 42
WIDTH_1 = 0.01
WIDTH_2 = 0.015
WIDTHS = {i: WIDTH_1 if (i in range(1, 11) or i in range(32, 38)) else WIDTH_2 for i in BLOCK_IDS}
RESTRICTED = [1,2,3,11,12,24,32,33,34,38,39,40,41,51,52]
X_GRID = np.round(np.arange(-0.2650, 0.1460, 0.001), 4)
EXTRA_POS = np.array([-0.2625, 0.1425])
X_GRID = np.unique(np.sort(np.concatenate([X_GRID, EXTRA_POS])))
POSSIBLE_GAPS = np.round(np.arange(0, 0.021, 0.001), 4)
SAVE_MEAN_BY = 0.045
SAVE_STD_BY = 5e-4
REFERENCE_GRID = None
BLOCK_XS = {}
history = {"gen": [], "best_mean": [], "best_hindex": [], "mean_mean": [], "mean_hindex": [], "valid_rate": [], "best_std": []}
best_solutions_table = []
CURRENT_GEN = 0
POP_SIZE = 10000
GENERATIONS = 1000
# =====================================================
# Seed Config IDs for initial population
# =====================================================
SEED_CONFIG_IDS = [965,2864,491,2137,3514,3793,21,1852,574,1793,
                   1595,5050,4018,1458,2409,3000,1760,4074,963,
                   4805,1533,375,2486,2459,4364,4835,335,2387,
                   2840,4153,1275,2267,3015,79,3433,1189,2330,
                   1403,2261]
# =====================================================
# Seed loader
# =====================================================
def load_seed_config(config_id):
    """
    Loads a seed configuration from CONFIG_DIR.
    Expects:
      - Config_<id>_fixedDim.csv : columns 'block' and 'x_pos_mm' (mm)
      - Config_<id>.csv : field map (x_mm,y_mm,z_mm,Bx_T,By_T,Bz_T)
    Returns encoded genome (b_array, x_array)
    """
    config_file = os.path.join(CONFIG_DIR, f"Config_{config_id}_fixedDim.csv")
    field_file = os.path.join(CONFIG_DIR, f"Config_{config_id}.csv")
    if not os.path.exists(config_file):
        raise RuntimeError(f"Config file {config_file} not found!")
    if not os.path.exists(field_file):
        raise RuntimeError(f"Field map file {field_file} not found!")
    df = pd.read_csv(config_file)
    # Expecting column names 'block' and 'x_pos_mm' as you indicated
    if 'block' not in df.columns or 'x_pos_mm' not in df.columns:
        raise RuntimeError(f"Seed config {config_file} missing required columns 'block' and 'x_pos_mm'")
    blocks = df['block'].astype(int).tolist()
    xs = [float(x)/1000.0 for x in df['x_pos_mm'].tolist()] # mm -> m
    # set REFERENCE_GRID if not already set
    global REFERENCE_GRID
    if REFERENCE_GRID is None:
        df_field = pd.read_csv(field_file, usecols=[0,1,2], dtype=np.float64)
        REFERENCE_GRID = df_field.copy()
    # encode into genome
    return encode_genome(blocks, xs)
# =====================================================
# FILE LOADING (cached per process)
# =====================================================
@lru_cache(maxsize=4096)
def load_field(block_id: int, x_val: float):
    """
    Returns np.ndarray (N_points, 3) columns (Bx, By, Bz) for a given block and center x.
    """
    global REFERENCE_GRID
    fname = os.path.join(BASE_DIR, f"C{int(block_id)}", f"x={x_val:+.4f}.csv")
    if not os.path.exists(fname):
        return None
    try:
        df_b = pd.read_csv(fname, usecols=[3,4,5], header=0, dtype=np.float32, engine='c')
        arr = df_b.values
        if REFERENCE_GRID is None:
            df_coords = pd.read_csv(fname, usecols=[0,1,2], header=0, dtype=np.float64, engine='c')
            REFERENCE_GRID = df_coords.copy()
        return arr
    except Exception as e:
        print(f"[load_field] Error reading {fname}: {e}")
        return None
def build_block_xs(block_id):
    """Cache available x positions for a given block folder."""
    if block_id in BLOCK_XS:
        return BLOCK_XS[block_id]
    folder = os.path.join(BASE_DIR, f"C{int(block_id)}")
    if not os.path.isdir(folder):
        BLOCK_XS[block_id] = np.array([], dtype=np.float64)
        return BLOCK_XS[block_id]
    files = [f for f in os.listdir(folder) if f.startswith("x=") and f.endswith(".csv")]
    xs = np.array(sorted([float(f[2:-4]) for f in files]), dtype=np.float64)
    BLOCK_XS[block_id] = xs
    return xs
def get_nearest_x(block_id, x):
    xs = build_block_xs(block_id)
    if xs.size == 0:
        return float(x) # fallback, will be validated elsewhere
    idx = np.argmin(np.abs(xs - x))
    return float(xs[idx])
# =====================================================
# VALID CONFIG GENERATOR
# =====================================================
def sample_valid_config(max_tries=1000000):
    """Generate a valid configuration by construction (first & last set, interior built greedily)."""
    for _ in range(max_tries):
        first_c = random.choice(BLOCK_IDS)
        w_first = WIDTHS[first_c]
        first_x = -0.2650 if w_first == WIDTH_1 else -0.2625
        last_c = random.choice(BLOCK_IDS)
        w_last = WIDTHS[last_c]
        last_x = 0.1450 if w_last == WIDTH_1 else 0.1425
        # restricted check
        if last_x > -0.1 and last_c in RESTRICTED:
            continue
        if first_x + w_first/2 >= last_x - w_last/2:
            continue
        selected_c = [first_c]
        selected_x = [first_x]
        current_edge = first_x + w_first/2
        max_center = last_x - w_last/2
        while current_edge < max_center - 0.01:
            c = random.choice(BLOCK_IDS)
            w = WIDTHS[c]
            gap = float(random.choice(POSSIBLE_GAPS))
            desired_center = current_edge + gap + w/2
            candidates = X_GRID[X_GRID >= desired_center]
            if len(candidates) == 0:
                break
            x = float(candidates[0])
            if x + w/2 > max_center:
                break
            if x > -0.1 and c in RESTRICTED:
                continue
            selected_c.append(c)
            selected_x.append(x)
            current_edge = x + w/2
        selected_c.append(last_c)
        selected_x.append(last_x)
        N = len(selected_c)
        if not (MIN_BLOCKS <= N <= MAX_BLOCKS):
            continue
        # snap to nearest available x filenames per block
        for j in range(N):
            selected_x[j] = get_nearest_x(selected_c[j], selected_x[j])
        # final check for gaps and restrictions before return
        ok = True
        for j in range(1, N):
            left_edge = selected_x[j-1] + WIDTHS[selected_c[j-1]]/2
            right_left = selected_x[j] - WIDTHS[selected_c[j]]/2
            gap = right_left - left_edge
            if gap < -1e-9 or gap > 0.0201:
                ok = False
                break
        if not ok:
            continue
        viol = False
        for xx, bb in zip(selected_x, selected_c):
            if xx > -0.1 and bb in RESTRICTED:
                viol = True; break
        if viol:
            continue
        return selected_c, selected_x
    raise RuntimeError("Could not generate valid initial configuration after many tries")
# =====================================================
# ENCODE / DECODE genome
# =====================================================
def encode_genome(blocks, xs):
    b = np.zeros(MAX_BLOCKS, dtype=np.int32)
    x = np.zeros(MAX_BLOCKS, dtype=np.float64)
    for i, (bb, xx) in enumerate(zip(blocks, xs)):
        b[i] = int(bb)
        x[i] = float(xx)
    return b, x
def decode_genome(b_arr, x_arr):
    mask = (b_arr != 0)
    blocks = [int(v) for v in b_arr[mask].tolist()]
    xs = [float(v) for v in x_arr[mask].tolist()]
    return blocks, xs
def repair_genome(b_arr, x_arr):
    """
    Adjust (do not reject) a genome to make it feasible:
      - sort by x
      - fix first and last to required centers
      - clamp interior centers to feasible ranges
      - snap centers to nearest available x values
      - ensure restricted blocks are moved left of -0.1 if needed
    Returns padded arrays (b_new, x_new).
    """
    b = b_arr.copy()
    x = x_arr.copy()
    blocks, xs = decode_genome(b, x)
    if len(blocks) == 0:
        return None
    # ---- Sort by x ----
    pairs = sorted(zip(xs, blocks), key=lambda p: p[0])
    xs = [p[0] for p in pairs]
    blocks = [p[1] for p in pairs]
    N = len(blocks)
    # ---- Fix first block ----
    w_first = WIDTHS[blocks[0]]
    xs[0] = -0.2650 if w_first == WIDTH_1 else -0.2625
    xs[0] = get_nearest_x(blocks[0], xs[0])
    # ---- Fix last block ----
    w_last = WIDTHS[blocks[-1]]
    xs[-1] = 0.1450 if w_last == WIDTH_1 else 0.1425
    xs[-1] = get_nearest_x(blocks[-1], xs[-1])
    # ---- Fix interior ----
    for i in range(1, N-1):
        left_edge = xs[i-1] + WIDTHS[blocks[i-1]]/2
        right_edge = xs[i+1] - WIDTHS[blocks[i+1]]/2
        w = WIDTHS[blocks[i]]
        min_center = left_edge + w/2
        max_center = right_edge - w/2
        # If impossible spacing → clamp inside feasible range (take midpoint)
        if min_center > max_center:
            xs[i] = 0.5 * (min_center + max_center)
        else:
            # clamp original position
            xs[i] = min(max(xs[i], min_center), max_center)
        xs[i] = get_nearest_x(blocks[i], xs[i])
        # Restricted check → shift left until valid (snap to nearest left grid)
        if xs[i] > -0.1 and blocks[i] in RESTRICTED:
            # try moving left to the largest allowed x <= -0.1000
            xs_avail = build_block_xs(blocks[i])
            if xs_avail.size > 0:
                left_options = xs_avail[xs_avail <= -0.1000]
                if left_options.size > 0:
                    xs[i] = float(left_options[-1])
                else:
                    # fallback: set to -0.1000 and snap
                    xs[i] = get_nearest_x(blocks[i], -0.1000)
            else:
                xs[i] = get_nearest_x(blocks[i], -0.1000)
    # ---- Final sanity & rebuild padded arrays ----
    b_new = np.zeros(MAX_BLOCKS, dtype=np.int32)
    x_new = np.zeros(MAX_BLOCKS, dtype=np.float64)
    for i in range(N):
        b_new[i] = int(blocks[i])
        x_new[i] = float(xs[i])
    return b_new, x_new
# =====================================================
# FITNESS / EVALUATION (multi-objective wrapper)
# =====================================================
def compute_h_index(By):
    mean_by = float(np.mean(By))
    return float(((By.max() - By.min()) / mean_by) * 1e6)
def evaluate_individual_multi(block_ids_bytes, x_positions_bytes):
    """
    Returns tuple (objective_value, h_ppm)
    objective_value uses -h_ppm + small penalties so that higher objective = better.
    If invalid, returns (-inf, inf).
    """
    try:
        b_arr = np.frombuffer(block_ids_bytes, dtype=np.int32, count=MAX_BLOCKS)
        x_arr = np.frombuffer(x_positions_bytes, dtype=np.float64, count=MAX_BLOCKS)
    except Exception:
        return -np.inf, np.inf
    blocks, xs = decode_genome(b_arr, x_arr)
    N = len(blocks)
    if N < MIN_BLOCKS or N > MAX_BLOCKS:
        return -np.inf, np.inf
    # validate first/last center
    w_first = WIDTHS[blocks[0]]
    expected_first = -0.2650 if w_first == WIDTH_1 else -0.2625
    if abs(xs[0] - expected_first) > 1e-6:
        return -np.inf, np.inf
    w_last = WIDTHS[blocks[-1]]
    expected_last = 0.1450 if w_last == WIDTH_1 else 0.1425
    if abs(xs[-1] - expected_last) > 1e-6:
        return -np.inf, np.inf
    # nearest x available check
    for j in range(N):
        nearest = get_nearest_x(blocks[j], xs[j])
        if abs(nearest - xs[j]) > 1e-9:
            return -np.inf, np.inf
    # gaps check
    for j in range(1, N):
        left_edge = xs[j-1] + WIDTHS[blocks[j-1]] / 2
        right_left = xs[j] - WIDTHS[blocks[j]] / 2
        gap = right_left - left_edge
        if gap < -1e-12 or gap > 0.02001 + 1e-12:
            return -np.inf, np.inf
    # restricted check
    for xx, bb in zip(xs, blocks):
        if xx > -0.1 and bb in RESTRICTED:
            return -np.inf, np.inf
    # load and sum fields
    B_total = None
    for bb, xx in zip(blocks, xs):
        arr = load_field(int(bb), float(xx))
        if arr is None:
            return -np.inf, np.inf
        if B_total is None:
            B_total = arr.copy()
        else:
            B_total += arr
    By = B_total[:, 1]
    mean_by = float(np.mean(By))
    h_ppm = compute_h_index(By)
    # Soft penalties and bounds on mean and transverse
    mean_abs_Bx = float(np.mean(np.abs(B_total[:,0])))
    mean_abs_Bz = float(np.mean(np.abs(B_total[:,2])))
    penalty = 0.0
    if mean_by < 0.05:
        penalty -= 0.01 * (0.05 - mean_by)
    if mean_by < 0.04:
        return -np.inf, np.inf
    if mean_abs_Bx >= 0.015 or mean_abs_Bz >= 0.02:
        penalty = -0.01*(mean_abs_Bx + mean_abs_Bz)
    if mean_abs_Bx >= 0.020 or mean_abs_Bz >= 0.05:
        return -np.inf, np.inf
        
    # dynamic threshold for h_ppm to keep population reasonable in early gens
#    threshold = 1000.0
#    if CURRENT_GEN < 801:
#        threshold = 5000.0
#    if CURRENT_GEN < 501:
#        threshold = 30000.0
#    if CURRENT_GEN < 51:
#        threshold = 100000.0
#    if h_ppm >= threshold:
#        return -np.inf, np.inf
#    return float(-h_ppm + penalty), float(h_ppm)
#def _eval_wrapper_multi(args):
#    return evaluate_individual_multi(*args)
    
# =====================================================
# SELECTION / CROSSOVER / MUTATION
# =====================================================
def tournament_selection(population, objectives, k=3):
    inds = np.random.choice(len(population), size=k, replace=False)
    best = inds[0]
    for i in inds:
        obj_i, h_i = objectives[i] # Adjusted to match (obj, h_ppm)
        obj_best, h_best = objectives[best]
        if obj_i > obj_best: # Higher objective better
            best = i
    return int(best)
def uniform_crossover_gap_safe(pa, pb, crossover_prob=1.0):
    """Uniform-style crossover with repair attempts to maintain validity."""
    if random.random() >= crossover_prob:
        return (pa[0].copy(), pa[1].copy()), (pb[0].copy(), pb[1].copy())
    n = MAX_BLOCKS
    mask = np.random.rand(n) < 0.5
    c1_b, c2_b = pa[0].copy(), pb[0].copy()
    c1_x, c2_x = pa[1].copy(), pb[1].copy()
    c1_b[mask], c2_b[mask] = c2_b[mask].copy(), c1_b[mask].copy()
    c1_x[mask], c2_x[mask] = c2_x[mask].copy(), c1_x[mask].copy()
    def fix_and_check(gen_b, gen_x):
        blocks, xs = decode_genome(gen_b, gen_x)
        N = len(blocks)
        if N < MIN_BLOCKS:
            return None
        # enforce ends
        w_first = WIDTHS[blocks[0]]
        xs[0] = get_nearest_x(blocks[0], -0.2650 if w_first == WIDTH_1 else -0.2625)
        w_last = WIDTHS[blocks[-1]]
        xs[-1] = get_nearest_x(blocks[-1], 0.1450 if w_last == WIDTH_1 else 0.1425)
        # try to re-place interior centers into allowed ranges
        for i in range(1, N-1):
            left_edge = xs[i-1] + WIDTHS[blocks[i-1]]/2
            right_edge = xs[i+1] - WIDTHS[blocks[i+1]]/2
            w = WIDTHS[blocks[i]]
            min_center = left_edge + w/2
            max_center = right_edge - w/2
            if min_center > max_center:
                continue
            xs_avail = build_block_xs(blocks[i])
            mask_local = (xs_avail >= min_center-1e-12) & (xs_avail <= max_center+1e-12)
            if np.any(mask_local):
                xs[i] = float(np.random.choice(xs_avail[mask_local]))
        for xx, bb in zip(xs, blocks):
            if xx > -0.1 and bb in RESTRICTED:
                return None
        out_b = np.zeros_like(gen_b); out_x = np.zeros_like(gen_x)
        for i, (bb, xx) in enumerate(zip(blocks, xs)):
            out_b[i] = int(bb); out_x[i] = float(xx)
        # Use adjust-style repair before returning
        repaired = repair_genome(out_b, out_x)
        if repaired is None:
            return None
        return repaired
    child1 = fix_and_check(c1_b, c1_x)
    child2 = fix_and_check(c2_b, c2_x)
    if child1 is None:
        child1 = (pa[0].copy(), pa[1].copy())
    if child2 is None:
        child2 = (pb[0].copy(), pb[1].copy())
    return child1, child2
def mutate_gap_aware(ind, gen, generations, p_block_base=0.15, p_pos_base=0.1, p_addrem_base=0.2):
    """
    Mutate while respecting gap restrictions.
    NOTE: insertion condition fixed so insertion is allowed unless it violates RESTRICTED rule.
    """
    p_block = p_block_base * (0.5 + 0.5*(1 - gen/generations))  # Declines less aggressively
    p_pos = p_pos_base * (0.5 + 0.5*(1 - gen/generations))
    p_addrem = p_addrem_base * (0.5 + 0.5*(1 - gen/generations))
    b = ind[0].copy(); x = ind[1].copy()
    blocks, xs = decode_genome(b, x)
    N = len(blocks)
    if N < MIN_BLOCKS:
        return (b, x)
    # change block type
    for i in range(N):
        if random.random() < p_block:
            newb = random.choice(BLOCK_IDS)
            if xs[i] > -0.1 and newb in RESTRICTED:
                continue
            blocks[i] = int(newb)
            xs[i] = get_nearest_x(blocks[i], xs[i])
            if i == 0:
                w_first = WIDTHS[blocks[0]]
                xs[0] = -0.2650 if w_first == WIDTH_1 else -0.2625
                xs[0] = get_nearest_x(blocks[0], xs[0])
            if i == N-1:
                w_last = WIDTHS[blocks[-1]]
                xs[-1] = 0.1450 if w_last == WIDTH_1 else 0.1425
                xs[-1] = get_nearest_x(blocks[-1], xs[-1])
    # move positions within allowed min/max
    for i in range(1, N-1):
        if random.random() < p_pos:
            left_edge = xs[i-1] + WIDTHS[blocks[i-1]]/2
            right_edge = xs[i+1] - WIDTHS[blocks[i+1]]/2
            w = WIDTHS[blocks[i]]
            min_center = left_edge + w/2
            max_center = right_edge - w/2
            if min_center > max_center:
                continue
            xs_avail = build_block_xs(blocks[i])
            mask_local = (xs_avail >= min_center-1e-12) & (xs_avail <= max_center+1e-12)
            if np.any(mask_local):
                xs[i] = float(np.random.choice(xs_avail[mask_local]))
    # insertion attempt
    if random.random() < p_addrem / 2 and N < MAX_BLOCKS:
        insert_idx = random.randint(1, N-1)
        new_c = random.choice(BLOCK_IDS)
        w = WIDTHS[new_c]
        left_edge = xs[insert_idx-1] + WIDTHS[blocks[insert_idx-1]] / 2
        right_edge = xs[insert_idx] - WIDTHS[blocks[insert_idx]] / 2
        min_center = left_edge + w/2
        max_center = right_edge - w/2
        if min_center < max_center:
            center = random.uniform(min_center, max_center)
            x_new = get_nearest_x(new_c, center)
            # FIX for insertion: allow insertion unless it violates restricted rule
            if not (x_new > -0.1 and new_c in RESTRICTED):
                blocks.insert(insert_idx, new_c)
                xs.insert(insert_idx, x_new)
                N += 1
    # removal attempt
    elif random.random() < p_addrem / 2 and N > MIN_BLOCKS:
        remove_idx = random.randint(1, N-2)
        del blocks[remove_idx]
        del xs[remove_idx]
        N -= 1
    # re-encode into fixed-length arrays
    b_new = np.zeros(MAX_BLOCKS, dtype=np.int32)
    x_new = np.zeros(MAX_BLOCKS, dtype=np.float64)
    for i, (bb, xx) in enumerate(zip(blocks, xs)):
        b_new[i] = int(bb); x_new[i] = float(xx)
    # Apply adjust-style repair to ensure endpoints and gaps
    repaired = repair_genome(b_new, x_new)
    if repaired is not None:
        return repaired
    return (b_new, x_new)
# =====================================================
# WORKER INIT
# =====================================================
def worker_init(seed_offset=0):
    import os, time, random as _r, numpy as _np
    seed = (int(time.time() * 1e6) ^ os.getpid() ^ (seed_offset or 0)) & 0xFFFFFFFF
    _r.seed(seed)
    _np.random.seed(seed % (2**32 - 1))
# =====================================================
# SAVE OUTPUTS (fix Generation field handling)
# =====================================================
def _compute_full_metrics(genome):
    blocks, xs = decode_genome(genome[0], genome[1])
    B_total = None
    for b, xx in zip(blocks, xs):
        arr = load_field(int(b), float(xx))
        if arr is None:
            return None
        if B_total is None:
            B_total = arr.copy()
        else:
            B_total += arr
    By = B_total[:, 1]
    mean_by = float(np.mean(By))
    std_by = float(np.std(By))
    h_ppm = compute_h_index(By)
    mean_abs_Bx = float(np.mean(np.abs(B_total[:,0])))
    mean_abs_Bz = float(np.mean(np.abs(B_total[:,2])))
    return dict(blocks=blocks, xs=xs, B_total=B_total, mean_by=mean_by, std_by=std_by, h_ppm=h_ppm,
                mean_abs_Bx=mean_abs_Bx, mean_abs_Bz=mean_abs_Bz)
def save_solution_if_qualifies(genome, metrics, tag_str, gen_int=None):
    """
    Save solution if metrics meet SAVE thresholds.
    tag_str -> used for filenames
    gen_int -> integer generation number (for the 'Generation' column).
    """
    if metrics is None:
        return False
    mean_by = metrics['mean_by']
    std_by = metrics['std_by']
    h_ppm = metrics['h_ppm']
    mean_abs_Bx = metrics['mean_abs_Bx']
    mean_abs_Bz = metrics['mean_abs_Bz']
    if mean_by <= SAVE_MEAN_BY or std_by >= SAVE_STD_BY:
        return False
    # field map (REFERENCE_GRID preserved order)
    field_df = REFERENCE_GRID.copy()
    field_df['Bx_T'] = metrics['B_total'][:,0]
    field_df['By_T'] = metrics['B_total'][:,1]
    field_df['Bz_T'] = metrics['B_total'][:,2]
    field_df.to_csv(os.path.join(OUTPUT_DIR, f"{tag_str}_field_map.csv"), index=False, float_format='%.7f')
    # padded config
    padded_b = np.pad(metrics['blocks'], (0, MAX_BLOCKS - len(metrics['blocks'])), constant_values=0)
    padded_x_mm = np.pad(np.array(metrics['xs']) * 1000.0, (0, MAX_BLOCKS - len(metrics['xs'])), constant_values=0.0)
    pd.DataFrame({'block': padded_b, 'x_pos_mm': padded_x_mm}).to_csv(os.path.join(OUTPUT_DIR, f"{tag_str}_config.csv"), index=False, float_format='%.3f')
    gen_value = gen_int if (isinstance(gen_int, int)) else -1
    params = pd.DataFrame([{
        'Generation': gen_value,
        'Mean_By_T': mean_by,
        'Std_By_T': std_by,
        'H_ppm': h_ppm,
        'Mean_abs_Bx_T': mean_abs_Bx,
        'Mean_abs_Bz_T': mean_abs_Bz,
        'N_blocks': len(metrics['blocks'])
    }])
    params.to_csv(os.path.join(OUTPUT_DIR, f"{tag_str}_params.csv"), index=False)
    best_solutions_table.append({'tag': tag_str, 'mean_by': mean_by, 'std_by': std_by, 'h_ppm': h_ppm,
                                 'mean_abs_Bx': mean_abs_Bx, 'mean_abs_Bz': mean_abs_Bz, 'n_blocks': len(metrics['blocks'])})
    return True
# =====================================================
# GA RUN LOOP
# =====================================================
def run_ga_multi(pop_size=POP_SIZE, generations=GENERATIONS, n_workers=3, seed=None, initial_population=None):
    global CURRENT_GEN
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    # --- FIX C: Load initial population from SEED_CONFIG_IDS if initial_population not provided ---
    if initial_population is None:
        initial_population = []
        for cfg_id in SEED_CONFIG_IDS:
            try:
                indiv = load_seed_config(cfg_id)
                initial_population.append(indiv)
            except Exception as e:
                print(f"[seed load] skipping {cfg_id}: {e}")
        # If loaded seeds exceed pop_size, trim
        if len(initial_population) > pop_size:
            initial_population = initial_population[:pop_size]
        # Fill remaining population with sampled valid configs, forcing length diversity
        target_lengths = np.random.randint(MIN_BLOCKS, MAX_BLOCKS+1, size=pop_size - len(initial_population))
        for target_n in target_lengths:
            tries = 0
            while tries < 100:
                c, x = sample_valid_config()
                if len(c) == target_n:  # Accept only if matches target to force variety
                    initial_population.append(encode_genome(c, x))
                    break
                tries += 1
            if tries == 100:  # Fallback to any valid if hard to match
                initial_population.append(encode_genome(*sample_valid_config()))
    else:
        # if initial_population provided (e.g. from saved state pickle), trust it but ensure correct length
        if len(initial_population) > pop_size:
            initial_population = initial_population[:pop_size]
        while len(initial_population) < pop_size:
            initial_population.append(encode_genome(*sample_valid_config()))
    population = initial_population.copy()
    # persistent process pool
    pool = Pool(n_workers, initializer=worker_init)
    best_genome = None
    best_obj = (-np.inf, np.inf)
    metrics_history = []
    best_overall_metrics = None
    best_overall_gen = -1
    stagnation_counter = 0
    previous_best_h = float('inf')
    try:
        for gen in range(generations):
            CURRENT_GEN = gen
            args = [(np.array(ind[0], copy=False).tobytes(),
                     np.array(ind[1], copy=False).tobytes()) for ind in population]
            results = list(tqdm(pool.imap(_eval_wrapper_multi, args),
                                total=pop_size, desc=f"Gen {gen}"))
            objectives = results # each element is (obj_value, h_ppm)
            # find valid indices
            valid_indices = [i for i, (obj, h) in enumerate(objectives) if np.isfinite(obj) and np.isfinite(h)]
            if len(valid_indices) > 0:
                best_idx = min(valid_indices, key=lambda i: objectives[i][1]) # smallest h_ppm
                best_h = objectives[best_idx][1]
            else:
                best_idx = None
                best_h = None
            print(f"Gen {gen}: Valid configs: {len(valid_indices)}/{pop_size}, Best H_ppm: {best_h}")
            # If valid candidate found -> compute full metrics and possibly save
            if best_idx is not None:
                best_obj = objectives[best_idx]
                best_genome = population[best_idx]
                metrics = _compute_full_metrics(best_genome)
                # ---- ALWAYS SAVE BEST OF GENERATION (proper indentation, no tabs) ----
                if metrics is not None:
                    tag = f"best_gen_gen{gen}"
                    # save config (unpadded) as requested: block + x_pos_mm (in mm)
                    pd.DataFrame({
                        'block': metrics['blocks'],
                        'x_pos_mm': np.array(metrics['xs']) * 1000.0
                    }).to_csv(os.path.join(OUTPUT_DIR, f"{tag}_config.csv"), index=False)
                    # save params requested
                    pd.DataFrame([{
                        'Generation': gen,
                        'H_ppm': metrics['h_ppm'],
                        'Mean_By_T': metrics['mean_by'],
                        'Std_By_T': metrics['std_by'],
                        'Mean_abs_Bx_T': metrics['mean_abs_Bx'],
                        'Mean_abs_Bz_T': metrics['mean_abs_Bz'],
                        'N_blocks': len(metrics['blocks'])
                    }]).to_csv(os.path.join(OUTPUT_DIR, f"{tag}_params.csv"), index=False)
                # continue original behavior
                if metrics is not None:
                    metrics_history.append(metrics)
                    print(f"Gen {gen}: Best mean(By)={metrics['mean_by']:.6f} T, Std(By)={metrics['std_by']:.2e} T, H(ppm)={metrics['h_ppm']:.2f}")
                    if best_overall_metrics is None or metrics['h_ppm'] < best_overall_metrics['h_ppm']:
                        best_overall_metrics = metrics
                        best_overall_gen = gen
                        # save with text tag but also pass integer generation to params
                        save_solution_if_qualifies(best_genome, metrics, f"best_overall_gen{gen}", gen_int=gen)
            # Check for stagnation
            if best_h is not None and best_h < previous_best_h:
                stagnation_counter = 0
                previous_best_h = best_h
            else:
                stagnation_counter += 1
            if stagnation_counter >= 5: # Changed from 2 to 5
                # Replace entire population with new samples for different N and configs
                population = [encode_genome(*sample_valid_config()) for _ in range(pop_size)]
                stagnation_counter = 0
                print(f"No improvement after 5 generations, resetting population to new configurations with potentially different N.")
            # --- FIX B: Freeze mutation/operators if no valid individuals this generation ---
            if len(valid_indices) == 0:
                # keep population unchanged (freeze)
                new_pop = [ (ind[0].copy(), ind[1].copy()) for ind in population ]
                print("No valid individuals this generation -> GA mutation/operators are frozen (population retained).")
            else:
                # normal GA progression
                new_pop = []
                # Sort by highest objective (equivalent to lowest cost/H_ppm)
                sorted_indices = sorted(valid_indices, key=lambda i: objectives[i][0], reverse=True)
                elites_num = int(pop_size * 0.01)  # Reduced from 0.05 to 0.01
                for i in sorted_indices[:elites_num]:
                    new_pop.append((population[i][0].copy(), population[i][1].copy()))
                remaining = pop_size - len(new_pop)
                crossover_fraction = 0.55
                crossover_num = int(remaining * crossover_fraction)
                mutation_num = remaining - crossover_num
                # Add crossover offspring (no mutation)
                for _ in range((crossover_num // 2) + (crossover_num % 2)):
                    p1 = tournament_selection(population, objectives)
                    p2 = tournament_selection(population, objectives)
                    c1, c2 = uniform_crossover_gap_safe(population[p1], population[p2], crossover_prob=1.0)
                    new_pop.append(c1)
                    if len(new_pop) < pop_size:
                        new_pop.append(c2)
                # Add mutation offspring
                for _ in range(mutation_num):
                    p = tournament_selection(population, objectives)
                    child = (population[p][0].copy(), population[p][1].copy())
                    child = mutate_gap_aware(child, gen=gen, generations=generations)
                    new_pop.append(child)
                # Trim if slightly over due to odd numbers
                new_pop = new_pop[:pop_size]
            population = new_pop
            gc.collect()
            # periodic plots of progress (every 10 gens)
            if gen % 10 == 0 and len(metrics_history) > 0:
                gens_plot = list(range(len(metrics_history)))
                h_vals = [m['h_ppm'] for m in metrics_history]
                mean_by_vals = [m['mean_by'] for m in metrics_history]
                plt.figure(figsize=(10,4))
                plt.plot(gens_plot, h_vals, '-o')
                plt.xlabel("Saved-valid-index"); plt.ylabel("H(ppm)"); plt.title("H-index (best valid) vs saved index")
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, f"GA_progress_H_savedindex_gen{gen}.png"))
                plt.close()
                plt.figure(figsize=(10,4))
                plt.plot(gens_plot, mean_by_vals, '-o')
                plt.xlabel("Saved-valid-index"); plt.ylabel("Mean(By) [T]"); plt.title("Mean(By) (best valid) vs saved index")
                plt.tight_layout()
                plt.savefig(os.path.join(OUTPUT_DIR, f"GA_progress_meanBy_savedindex_gen{gen}.png"))
                plt.close()
    except KeyboardInterrupt:
        print("\n\n===== MANUAL STOP DETECTED (Ctrl+C) =====")
        if best_overall_metrics is not None:
            save_solution_if_qualifies(best_genome, metrics, f"interrupted_best", gen_int=best_overall_gen)
        return best_genome, best_obj
    pool.close()
    pool.join()
    # Final summary + final save using integer generation if available
    if best_overall_metrics is not None:
        save_solution_if_qualifies(best_genome, best_overall_metrics, "best_final", gen_int=best_overall_gen)
        print("\nBest metrics (minimum H-index) found:")
        print(f"Gen: {best_overall_gen}")
        print(f"H(ppm): {best_overall_metrics['h_ppm']:.2f}")
        print(f"Mean(By): {best_overall_metrics['mean_by']:.6f} T")
        print(f"Mean(|Bx|): {best_overall_metrics['mean_abs_Bx']:.6f} T")
        print(f"Mean(|Bz|): {best_overall_metrics['mean_abs_Bz']:.6f} T")
    # Save state for extension
    with open('ga_state.pkl', 'wb') as f:
        pickle.dump((population, best_genome, best_obj), f)
    return best_genome, best_obj
# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    TARGET_H = 10000.0
    INITIAL_GENS = 1000
    EXTRA_GENS = 100
    total_gens = 0
    best_H = float("inf")
    best_genome = None
    best_obj = None
    start = time.time()
    initial_pop = None
    if os.path.exists('ga_state.pkl'):
        try:
            with open('ga_state.pkl', 'rb') as f:
                initial_pop, _, _ = pickle.load(f)
        except Exception as e:
            print(f"[startup] failed to load ga_state.pkl: {e}")
            initial_pop = None
    print(f"\n=== Running initial {INITIAL_GENS} generations ===\n")
    best_genome, best_obj = run_ga_multi(
        pop_size=POP_SIZE,
        generations=INITIAL_GENS,
        n_workers=max(1, min(3, cpu_count()-1)),
        seed=None,
        initial_population=initial_pop
    )
    # If best_obj hasn't been updated to a valid objective, try to extract H from best_overall saved files
    try:
        best_H = best_obj[1]
    except Exception:
        best_H = float("inf")
    total_gens += INITIAL_GENS
    print(f"\nAfter {total_gens} generations: Best H = {best_H}")
    # continue by chunks until target achieved
    while best_H > TARGET_H:
        print(f"\n=== H = {best_H:.2f} > {TARGET_H}, extending by {EXTRA_GENS} generations ===\n")
        # attempt to reload last population from state file if exists
        if os.path.exists('ga_state.pkl'):
            try:
                with open('ga_state.pkl', 'rb') as f:
                    initial_pop, _, _ = pickle.load(f)
            except Exception:
                initial_pop = None
        best_genome, best_obj = run_ga_multi(
            pop_size=POP_SIZE,
            generations=EXTRA_GENS,
            n_workers=max(1, min(3, cpu_count()-1)),
            seed=None,
            initial_population=initial_pop
        )
        try:
            best_H = best_obj[1]
        except Exception:
            best_H = float("inf")
        total_gens += EXTRA_GENS
        print(f"After {total_gens} generations: Best H = {best_H}")
    elapsed = time.time() - start
    print(f"\n=== GA FINISHED: Goal achieved ===")
    print(f"Total generations = {total_gens}")
    print(f"Best H = {best_H}")
    print(f"Time = {elapsed/60:.2f} minutes")