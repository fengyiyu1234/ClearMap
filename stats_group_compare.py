# -*- coding: utf-8 -*-
"""
Group comparison statistics across the CCF ontology hierarchy.

Compares cell counts (Count / Percentage / Density / Volume) between two
user-defined groups of samples, for every cell class found under each sample's
`cell_registration/<class_name>/cell_registration.csv`, at every level of the
Allen CCF ontology tree (root = level 0, finer subregions = higher levels).

Statistical test: Welch's t-test (unequal variance), FDR-corrected (Benjamini-
Hochberg) separately within each ontology level, since the number of regions
tested (the hypothesis family size) varies hugely by level.

Usage:
    python stats_group_compare.py --config stats_config.yaml
"""
import os
import re
import argparse

import numpy as np
import pandas as pd
import yaml
from scipy import stats

import ClearMap.Alignment.Annotation as ano
from ClearMap.Analysis.Statistics.MultipleComparisonCorrection import correct_p_values
from ClearMap.IO import IO as clearmap_io
from ClearMap.IO import MHD as clearmap_mhd

BACKGROUND_NAMES = {'background', 'no label', ''}
METRICS = ('Count', 'Percentage', 'Density', 'Volume')


# ================= Config =================

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# ================= Cell registration CSV parsing =================

def clean_name(raw):
    """Strip numpy's byte-repr wrapper, e.g. b'Cerebral cortex' -> Cerebral cortex."""
    s = str(raw).strip()
    s = re.sub(r"^b['\"]", "", s)
    s = re.sub(r"['\"]$", "", s)
    return s.strip()


def read_cell_registration(csv_path):
    """cell_registration.csv has no header; columns are (in order):
    x, y, z, xr, yr, zr, xt, yt, zt, graph_order, name. Over-allocate column
    names since some rows can be ragged (trailing commas etc.)."""
    df = pd.read_csv(csv_path, header=None, names=range(20), engine='python')
    graph_order = pd.to_numeric(df[9], errors='coerce').fillna(-1).astype(int).values
    names = df[10].map(clean_name).values
    return graph_order, names


def filter_valid(graph_order, names):
    """Drop background/unmapped cells. Must filter by name, not graph_order==0,
    because unmapped cells are written with graph_order=0 which numerically
    collides with the ontology root region's own real graph_order=0."""
    lowered = np.array([str(n).strip().lower() for n in names])
    mask = ~np.isin(lowered, list(BACKGROUND_NAMES)) & (graph_order >= 0)
    return graph_order[mask]


def class_counts_for_sample(csv_path):
    """Returns (bins, total_valid). bins is a hierarchical rollup: bins[order]
    = count directly in that region + all its descendants."""
    if not os.path.exists(csv_path):
        print(f"  [warn] missing {csv_path}, treating as zero cells")
        return np.zeros(ano.n_structures, dtype=float), 0
    try:
        graph_order, names = read_cell_registration(csv_path)
    except pd.errors.EmptyDataError:
        return np.zeros(ano.n_structures, dtype=float), 0

    valid = filter_valid(graph_order, names)
    if len(valid) == 0:
        return np.zeros(ano.n_structures, dtype=float), 0

    # Convert to ClearMap's dense 'order' index first, then roll up hierarchically.
    # NOTE: ano.count_label() has a bug if called with any key other than 'order'
    # (it passes an unsupported `invalid=` kwarg to convert_label internally), so
    # always pre-convert to 'order' ourselves and call count_label(key='order').
    order_arr = ano.convert_label(valid, key='graph_order', value='order')
    bins = ano.count_label(order_arr, key='order', hierarchical=True).astype(float)
    return bins, len(valid)


# ================= Ontology metadata & region volumes =================

def normalize_class_key(name):
    """Token-set key for fuzzy class-folder matching: lowercase, split on
    non-alphanumerics, drop pure-numeric tokens (batch/version markers like
    the '3' in 'glia_3_GFP') so naming drift across samples -- e.g.
    'glia_3_GFP' vs 'glia_GFP' -- still resolves to the same class."""
    tokens = re.split(r'[^a-zA-Z0-9]+', name)
    return frozenset(t.lower() for t in tokens if t and not t.isdigit())


def list_sample_classes(data_dir, sample):
    reg_dir = os.path.join(data_dir, sample, 'cell_registration')
    if not os.path.isdir(reg_dir):
        return []
    return sorted(d for d in os.listdir(reg_dir) if os.path.isdir(os.path.join(reg_dir, d)))


def resolve_sample_class_dir(data_dir, sample, class_name, _cache={}):
    """Find the actual cell_registration/<...> folder for `class_name` within
    one sample, tolerating naming drift (see normalize_class_key). Returns
    None if no folder matches."""
    if sample not in _cache:
        _cache[sample] = list_sample_classes(data_dir, sample)
    actual = _cache[sample]

    if class_name in actual:
        return class_name

    key = normalize_class_key(class_name)
    matches = [d for d in actual if normalize_class_key(d) == key]
    if not matches:
        return None
    if len(matches) > 1:
        print(f"  [warn] ambiguous class match for '{class_name}' in {sample}: {matches}; using '{matches[0]}'")
    return matches[0]


def discover_classes(data_dir, sample_names, explicit=None):
    """Returns canonical class labels. When auto-discovering (explicit=None),
    folder names found across samples are grouped by normalize_class_key so
    that differently-named variants of the same marker combination (e.g.
    'glia_3_GFP' on one sample, 'glia_GFP' on another) collapse into one
    class instead of being treated as two."""
    if explicit:
        return list(explicit)

    all_names = sorted({name for s in sample_names for name in list_sample_classes(data_dir, s)})
    groups = {}
    for name in all_names:
        groups.setdefault(normalize_class_key(name), []).append(name)
    # shortest (then alphabetically first) name in each group becomes the label
    canonical = [min(names, key=lambda n: (len(n), n)) for names in groups.values()]
    return sorted(canonical)


def compute_combined_categories(counts_df, totals_df, category_formulas, classes):
    """Compute biologically-meaningful aggregate categories as a *signed*
    (+/-) combination of existing classes, e.g.:
        glia_total = glia_GFP + glia_RFP + glia_GFP_RFP + ...

    Our marker-combination classes come from brain_detector's upstream
    cross-channel matching (stitcher.py: match_soma_3d_iou + _merge_class,
    annotate_soma_with_tf_containment), which greedily matches candidate
    cells across marker channels via 3D IoU/containment and merges each
    matched group into ONE row with ONE composite class label before the
    per-class CSVs are written. So each physical cell lands in exactly one
    class file -- these classes are mutually-exclusive partitions, not
    nested/cumulative subsets -- and combining them is just a plain sum
    (every term's sign is "+" in practice). The signed-term support exists
    so this same engine still works correctly if some other dataset's
    classes DO nest (where a "-" term would be needed to avoid double
    counting); nothing here assumes one or the other.

    category_formulas: dict name -> list of {"sign": "+"/"-", "class": name}
    terms. Component class names are matched against `classes` the same
    fuzzy way as sample folders (see normalize_class_key).

    This is exact at every ontology level because hierarchical rollup is a
    linear operator: rollup(A) + rollup(B) - rollup(C) == rollup(A + B - C)
    for every region, so doing the +/- arithmetic directly on the already
    hierarchically-rolled-up 'count' column is equivalent to doing it on the
    raw (pre-rollup) cell sets.

    Returns (combined_counts_df, combined_totals_df, resolved_info) where
    resolved_info maps name -> a human-readable resolved formula string (for
    documentation/traceability). These are returned as separate DataFrames
    (not concatenated onto counts_df/totals_df) so callers can keep the raw
    per-class analysis untouched and route combined categories elsewhere.
    """
    resolved_info = {}
    empty_counts = counts_df.iloc[0:0]
    empty_totals = totals_df.iloc[0:0]
    if not category_formulas:
        return empty_counts, empty_totals, resolved_info

    combined_count_frames = []
    combined_total_frames = []
    for name, terms in category_formulas.items():
        resolved_terms = []
        for term in terms:
            sign = 1 if str(term.get('sign', '+')).strip() == '+' else -1
            c = term['class']
            if c in classes:
                resolved_terms.append((sign, c))
                continue
            key = normalize_class_key(c)
            match = next((cl for cl in classes if normalize_class_key(cl) == key), None)
            if match is None:
                print(f"  [warn] combined category '{name}': component '{c}' not found among "
                      f"classes {classes}, skipping it")
                continue
            resolved_terms.append((sign, match))

        if not resolved_terms:
            print(f"  [warn] combined category '{name}': no valid components found, skipping "
                  f"this category entirely")
            continue

        weight_by_class = {}
        for sign, cls in resolved_terms:
            weight_by_class[cls] = weight_by_class.get(cls, 0) + sign

        sub_counts = counts_df[counts_df['class_name'].isin(weight_by_class)].copy()
        sub_counts['count'] = sub_counts['count'] * sub_counts['class_name'].map(weight_by_class)
        summed = sub_counts.groupby(['group', 'sample', 'order'], as_index=False)['count'].sum()
        summed['class_name'] = name
        combined_count_frames.append(summed[['group', 'sample', 'class_name', 'order', 'count']])

        sub_totals = totals_df[totals_df['class_name'].isin(weight_by_class)].copy()
        sub_totals['total_valid'] = sub_totals['total_valid'] * sub_totals['class_name'].map(weight_by_class)
        summed_totals = sub_totals.groupby(['group', 'sample'], as_index=False)['total_valid'].sum()
        summed_totals['class_name'] = name
        combined_total_frames.append(summed_totals[['group', 'sample', 'class_name', 'total_valid']])

        parts = []
        for i, (sign, cls) in enumerate(resolved_terms):
            if i == 0:
                parts.append(cls if sign > 0 else f"-{cls}")
            else:
                parts.append(f"+ {cls}" if sign > 0 else f"- {cls}")
        resolved_info[name] = ' '.join(parts)

    combined_counts_df = pd.concat(combined_count_frames, ignore_index=True) if combined_count_frames else empty_counts
    combined_totals_df = pd.concat(combined_total_frames, ignore_index=True) if combined_total_frames else empty_totals
    return combined_counts_df, combined_totals_df, resolved_info


def build_region_metadata():
    """One row per 'order' index (0..n_structures-1) with id/name/level."""
    return pd.DataFrame({
        'order': np.arange(ano.n_structures),
        'id': ano.get_list('id'),
        'graph_order': ano.get_list('graph_order'),
        'name': ano.get_list('name'),
        'level': ano.get_list('level'),
    })


def build_region_volumes(annotation_volume_path, voxel_size_um, cache_path=None, force=False):
    """Static per-region voxel/volume lookup computed once from the master
    atlas volume (not any per-sample cropped/oriented copy), so Density is
    comparable across samples regardless of each sample's own registration
    crop. Cached to disk since bincounting ~10^8 voxels is not free."""
    if cache_path and os.path.exists(cache_path) and not force:
        print(f"Loaded region volumes from cache: {cache_path}")
        return pd.read_csv(cache_path)

    print(f"Computing region volumes from {annotation_volume_path} (one-time)...")
    vol = clearmap_io.read(annotation_volume_path)
    ids_flat = np.asarray(vol).ravel().astype(int)
    # id == 0 means background/outside-brain (not a real Allen structure id -- the
    # ontology's own root id is 997) and must be excluded before lookup, same as the
    # background-cell filtering in filter_valid().
    valid_ids = ids_flat[ids_flat > 0]
    order_flat = ano.convert_label(valid_ids, key='id', value='order')
    voxel_bins = ano.count_label(order_flat, key='order', hierarchical=True).astype(float)
    voxel_volume_mm3 = float(np.prod(voxel_size_um)) * 1e-9  # um^3 -> mm^3

    df = pd.DataFrame({
        'order': np.arange(ano.n_structures),
        'voxel_count': voxel_bins,
        'volume_mm3': voxel_bins * voxel_volume_mm3,
    })
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        df.to_csv(cache_path, index=False)
    return df


def build_per_sample_region_volumes(data_dir, samples, voxel_size_um, master_volumes,
                                     cache_path=None, force=False):
    """Per-(sample, order) region volume in mm^3, from each sample's own
    transformix-warped atlas (<data_dir>/<sample>/volume/result.mhd) rather
    than the static master atlas used by build_region_volumes() -- so
    Density/Volume can reflect each sample's real anatomy (e.g. atrophy)
    instead of a group-invariant constant. The warped volume is nearest-
    neighbor interpolated (label-safe) and lives on that sample's own
    known-voxel-size grid, so voxel-counting per label is a standard
    atlas-based volumetry estimate (no Jacobian correction needed).

    Falls back to master_volumes (the region-only, sample-independent
    volumes) for any sample missing volume/result.mhd, with a warning, so
    every (order, sample) pair is always present in the returned table."""
    if cache_path and os.path.exists(cache_path) and not force:
        print(f"Loaded per-sample region volumes from cache: {cache_path}")
        return pd.read_csv(cache_path)

    voxel_volume_mm3 = float(np.prod(voxel_size_um)) * 1e-9  # um^3 -> mm^3
    # transformix's nearest-neighbor resampling of the master atlas into each
    # sample's native grid can, at region-boundary voxels, pick up a handful of
    # stray label values that aren't real ontology ids (interpolation artifact
    # -- the un-warped master atlas doesn't have this issue). Treat those like
    # background rather than crashing on an unknown id.
    known_ids = set(int(i) for i in ano.get_list('id'))

    frames = []
    for sample in samples:
        mhd_path = os.path.join(data_dir, sample, 'volume', 'result.mhd')
        if not os.path.exists(mhd_path):
            print(f"  [warn] missing {mhd_path}, falling back to master-atlas volumes for sample '{sample}'")
            frames.append(pd.DataFrame({
                'sample': sample,
                'order': master_volumes['order'].values,
                'voxel_count': master_volumes['voxel_count'].values,
                'volume_mm3': master_volumes['volume_mm3'].values,
            }))
            continue

        print(f"Computing per-sample region volumes for '{sample}' from {mhd_path} ...")
        vol = clearmap_mhd.mhd_read(mhd_path)
        ids_flat = np.asarray(vol).ravel().astype(int)
        # id == 0 means background/outside-brain, same convention as build_region_volumes().
        valid_ids = ids_flat[ids_flat > 0]
        known_mask = np.isin(valid_ids, list(known_ids))
        n_unknown = (~known_mask).sum()
        if n_unknown:
            print(f"  [warn] sample '{sample}': dropping {n_unknown} voxels "
                  f"({100 * n_unknown / len(valid_ids):.3f}%) with warp-interpolation-artifact ids "
                  f"not in the ontology: {sorted(set(valid_ids[~known_mask].tolist()))}")
        valid_ids = valid_ids[known_mask]
        order_flat = ano.convert_label(valid_ids, key='id', value='order')
        voxel_bins = ano.count_label(order_flat, key='order', hierarchical=True).astype(float)

        frames.append(pd.DataFrame({
            'sample': sample,
            'order': np.arange(ano.n_structures),
            'voxel_count': voxel_bins,
            'volume_mm3': voxel_bins * voxel_volume_mm3,
        }))

    df = pd.concat(frames, ignore_index=True)
    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        df.to_csv(cache_path, index=False)
    return df


# ================= Count collection =================

def collect_counts(data_dir, classes, group_samples):
    """group_samples: dict[group_key] -> list[sample names].
    Returns (counts_df, totals_df) long-format DataFrames."""
    n = ano.n_structures
    orders = np.arange(n)
    count_frames = []
    total_rows = []
    for group_key, samples in group_samples.items():
        for sample in samples:
            for cls in classes:
                actual_dir = resolve_sample_class_dir(data_dir, sample, cls)
                if actual_dir is None:
                    print(f"  [warn] no folder matching class '{cls}' for sample {sample}, treating as zero cells")
                    bins, total = np.zeros(n, dtype=float), 0
                else:
                    csv_path = os.path.join(data_dir, sample, 'cell_registration', actual_dir, 'cell_registration.csv')
                    bins, total = class_counts_for_sample(csv_path)
                count_frames.append(pd.DataFrame({
                    'group': group_key, 'sample': sample, 'class_name': cls,
                    'order': orders, 'count': bins,
                }))
                total_rows.append({'group': group_key, 'sample': sample, 'class_name': cls, 'total_valid': total})
    counts_df = pd.concat(count_frames, ignore_index=True)
    totals_df = pd.DataFrame(total_rows)
    return counts_df, totals_df


def metric_matrix(counts_df, totals_df, per_sample_volumes, class_name, metric, group_key, samples):
    """Returns a (n_structures, n_samples) matrix, row-ordered by 'order',
    column-ordered by `samples`."""
    sub = counts_df[(counts_df['class_name'] == class_name) & (counts_df['group'] == group_key)]
    pivot = sub.pivot(index='order', columns='sample', values='count').reindex(columns=samples)
    mat = pivot.values.astype(float)

    if metric == 'Count':
        return mat

    if metric == 'Percentage':
        tot = (totals_df[(totals_df['class_name'] == class_name) & (totals_df['group'] == group_key)]
               .set_index('sample')['total_valid'].reindex(samples).values.astype(float))
        pct = np.full_like(mat, np.nan)
        nonzero = tot > 0
        pct[:, nonzero] = mat[:, nonzero] / tot[nonzero] * 100.0
        return pct

    if metric in ('Density', 'Volume'):
        # Per-(order, sample) volume from each sample's own warped atlas, NOT a
        # region-only constant -- this is what lets Density reflect real
        # per-sample anatomy (e.g. atrophy) instead of being a fixed rescaling
        # of Count.
        vol_mat = (per_sample_volumes.pivot(index='order', columns='sample', values='volume_mm3')
                   .reindex(index=pivot.index, columns=samples).values.astype(float))
        if metric == 'Volume':
            return vol_mat
        density = np.full_like(mat, np.nan)
        nonzero = vol_mat > 0
        density[nonzero] = mat[nonzero] / vol_mat[nonzero]
        return density

    raise ValueError(f"Unknown metric: {metric}")


# ================= Statistics =================

def welch_ttest(mat_a, mat_b):
    """Welch's t-test (unequal variance), NaN p-values (e.g. zero-variance
    constant groups) treated as non-significant (p=1.0)."""
    with np.errstate(invalid='ignore'):
        _, p = stats.ttest_ind(mat_a, mat_b, axis=1, equal_var=False, nan_policy='omit')
    p = np.asarray(p, dtype=float)
    p[np.isnan(p)] = 1.0
    return p


def run_level_tests(mat_a, mat_b, metadata, level, class_name, metric, samples_a, samples_b):
    """Restrict to regions at this ontology level, drop regions with zero
    total signal in both groups, run Welch's t-test, then FDR-correct
    (BH) within this level's surviving p-values only (each level is its own
    hypothesis-testing family)."""
    idx = np.where((metadata['level'] == level).values)[0]
    if len(idx) == 0:
        return pd.DataFrame()

    a = mat_a[idx]
    b = mat_b[idx]
    total = np.nansum(a, axis=1) + np.nansum(b, axis=1)
    keep = total > 0
    idx, a, b = idx[keep], a[keep], b[keep]
    if len(idx) == 0:
        return pd.DataFrame()

    mean_a = np.nanmean(a, axis=1)
    mean_b = np.nanmean(b, axis=1)
    mean_a_is_zero = mean_a == 0
    with np.errstate(divide='ignore', invalid='ignore'):
        fold_change = mean_b / mean_a
        log2fc = np.log2(fold_change)
    fold_change = np.where(mean_a_is_zero, np.nan, fold_change)
    log2fc = np.where(mean_a_is_zero, np.nan, log2fc)
    # complete disappearance (mean_b==0) gives log2fc=-inf; not Excel-writable, so drop to NaN
    log2fc[np.isinf(log2fc)] = np.nan

    p = welch_ttest(a, b)
    p_fdr = correct_p_values(p, method='BH')

    out = metadata.iloc[idx].reset_index(drop=True).copy()
    out['class_name'] = class_name
    out['metric'] = metric
    out['n_a'] = a.shape[1]
    out['n_b'] = b.shape[1]
    out['mean_a'] = mean_a
    out['mean_b'] = mean_b
    out['fold_change'] = fold_change
    out['log2fc'] = log2fc
    out['mean_a_is_zero'] = mean_a_is_zero
    out['p_value'] = p
    out['p_fdr'] = p_fdr
    # Raw per-sample values behind mean_a/mean_b, one column per sample
    # (named after the sample itself), group A then group B.
    for i, s in enumerate(samples_a):
        out[s] = a[:, i]
    for j, s in enumerate(samples_b):
        out[s] = b[:, j]
    return out


# ================= Orchestration =================

def run_group_comparison(counts_df, totals_df, per_sample_volumes, volumes, metadata, class_names,
                          samples_a, samples_b):
    """Runs metric_matrix + run_level_tests across all levels/metrics for the
    given class_names, returning one concatenated, volume-annotated
    DataFrame (empty if class_names is empty or nothing survives the
    per-level zero-total filter in run_level_tests)."""
    max_level = int(metadata['level'].max())
    rows = []
    for cls in class_names:
        for metric in METRICS:
            mat_a = metric_matrix(counts_df, totals_df, per_sample_volumes, cls, metric, 'a', samples_a)
            mat_b = metric_matrix(counts_df, totals_df, per_sample_volumes, cls, metric, 'b', samples_b)
            for level in range(max_level + 1):
                res = run_level_tests(mat_a, mat_b, metadata, level, cls, metric, samples_a, samples_b)
                if not res.empty:
                    rows.append(res)

    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not result.empty:
        result[['fold_change', 'log2fc']] = result[['fold_change', 'log2fc']].replace([np.inf, -np.inf], np.nan)
        # Static master-atlas reference volume only -- NOT what Density/Volume
        # metric rows' mean_a/mean_b are computed from (those use each sample's
        # own warped-atlas volume; see per_sample_volumes).
        result = result.merge(
            volumes[['order', 'volume_mm3']].rename(columns={'volume_mm3': 'volume_mm3_atlas_ref'}),
            on='order', how='left')
    return result


def run_all_stats(cfg):
    ano.initialize(label_file=cfg.get('label_file'))
    metadata = build_region_metadata()

    annotation_volume = cfg.get('annotation_volume') or ano.default_vol_annotation_file
    voxel_size_um = cfg.get('voxel_size_um', [20.0, 20.0, 20.0])
    volume_cache = cfg.get('volume_cache')
    volumes = build_region_volumes(annotation_volume, voxel_size_um, volume_cache)

    groups_cfg = cfg['groups']
    group_a_name = groups_cfg['a'].get('name', 'A')
    group_b_name = groups_cfg['b'].get('name', 'B')
    samples_a = groups_cfg['a']['samples']
    samples_b = groups_cfg['b']['samples']

    all_samples = sorted(set(samples_a) | set(samples_b))
    per_sample_volumes = build_per_sample_region_volumes(
        cfg['data_dir'], all_samples, voxel_size_um, volumes, cfg.get('per_sample_volume_cache'))

    classes = cfg.get('classes') or discover_classes(cfg['data_dir'], samples_a + samples_b)
    print(f"Classes: {classes}")
    print(f"Group A ({group_a_name}): {samples_a}")
    print(f"Group B ({group_b_name}): {samples_b}")

    counts_df, totals_df = collect_counts(cfg['data_dir'], classes, {'a': samples_a, 'b': samples_b})

    result = run_group_comparison(counts_df, totals_df, per_sample_volumes, volumes, metadata,
                                   classes, samples_a, samples_b)
    if not result.empty:
        result['formula'] = ''

    combined_counts_df, combined_totals_df, formula_info = compute_combined_categories(
        counts_df, totals_df, cfg.get('combined_categories'), classes)
    if formula_info:
        print(f"Combined categories: {formula_info}")
        combined_result = run_group_comparison(
            combined_counts_df, combined_totals_df, per_sample_volumes, volumes, metadata,
            list(formula_info.keys()), samples_a, samples_b)
        if not combined_result.empty:
            combined_result['formula'] = combined_result['class_name'].map(formula_info)
            # Combined categories are folded straight into the same rows/sheets
            # as the raw classes (distinguished only by the 'formula' column),
            # not routed to a separate sheet.
            result = pd.concat([result, combined_result], ignore_index=True) if not result.empty else combined_result

    return (result, group_a_name, group_b_name, counts_df, totals_df, volumes, per_sample_volumes, metadata,
            combined_counts_df, combined_totals_df, formula_info)


def write_outputs(df, cfg, group_a_name, group_b_name):
    out_cfg = cfg.get('output', {})
    out_dir = out_cfg.get('dir', './stats_output')
    os.makedirs(out_dir, exist_ok=True)

    long_table_path = os.path.join(out_dir, out_cfg.get('long_table', 'region_stats.csv'))
    df.to_csv(long_table_path, index=False)
    print(f"Wrote long-format table: {long_table_path} ({len(df)} rows)")

    excel_path = os.path.join(out_dir, out_cfg.get('excel', 'region_stats_by_level.xlsx'))
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        readme = pd.DataFrame({
            'column': ['level', 'order', 'id', 'graph_order', 'name', 'class_name', 'formula', 'metric',
                       'n_a', 'n_b', 'mean_a', 'mean_b', 'fold_change', 'log2fc',
                       'mean_a_is_zero', 'p_value', 'p_fdr', 'volume_mm3_atlas_ref', '<sample name>'],
            'description': [
                'Ontology tree depth from root (0=whole brain, higher=finer subregions)',
                "ClearMap's internal dense region index (stable within one ontology file)",
                'Allen CCF structure id',
                "Allen ontology 'graph_order' value",
                'Region name',
                'Cell class / marker-combination name (from cell_registration/<class_name>/ folder), '
                'or an aggregate category name defined in config combined_categories (e.g. glia_total)',
                "Resolved +/- formula over raw classes, for aggregate categories only "
                '(empty for ordinary, non-aggregate classes)',
                'Count / Percentage / Density / Volume',
                f'Number of samples in group A ({group_a_name})',
                f'Number of samples in group B ({group_b_name})',
                f'Mean metric value, group A ({group_a_name}) -- for Volume, mean region volume in mm^3 '
                "from each sample's own transformix-warped atlas (see per-sample volumes, not "
                'volume_mm3_atlas_ref)',
                f'Mean metric value, group B ({group_b_name}); see mean_a note',
                'mean_b / mean_a (NaN if mean_a == 0 -- fold change undefined)',
                'log2(fold_change); NaN if undefined or infinite (complete presence/absence)',
                'True if mean_a == 0 (fold_change/log2fc undefined for this region)',
                "Welch's t-test p-value (unequal variance), raw/uncorrected",
                'Benjamini-Hochberg FDR-corrected p-value, computed separately within each level',
                'Static reference only: region volume in mm^3 from the master atlas, same for every '
                'sample/group. NOT what Density or Volume metric rows are computed from -- those use '
                "each sample's own warped-atlas volume, which varies per sample (e.g. with atrophy).",
                'One column per sample (named after the sample folder), group A samples first then '
                "group B, matching the n_a/n_b order. Holds that sample's own raw value for this "
                "row's metric (same unit as mean_a/mean_b) -- the individual values behind each mean.",
            ],
        })
        readme.to_excel(writer, sheet_name='ReadMe', index=False)

        volumes_out = df[['order', 'id', 'name', 'level', 'volume_mm3_atlas_ref']].drop_duplicates().sort_values('order')
        volumes_out.to_excel(writer, sheet_name='Region_Volumes', index=False)

        max_level = int(df['level'].max()) if not df.empty else -1
        for level in range(max_level + 1):
            sheet = df[df['level'] == level].sort_values(['class_name', 'metric', 'p_fdr'])
            if sheet.empty:
                continue
            sheet.to_excel(writer, sheet_name=f'L{level:02d}', index=False)
    print(f"Wrote Excel workbook: {excel_path}")


def write_per_sample_tables(counts_df, totals_df, volumes, per_sample_volumes, metadata, combined_counts_df,
                             combined_totals_df, formula_info, out_dir):
    """One workbook per sample, one sheet per ontology level (L00, L01, ...):
    every class (incl. combined categories) x every ontology region, with
    Count/Percentage/Density -- a full per-sample registration inventory,
    independent of any group comparison. Unlike the group-comparison output,
    zero-count regions are kept so the file can double as a QC view of one
    sample's whole registration."""
    sample_dir = os.path.join(out_dir, 'per_sample')
    os.makedirs(sample_dir, exist_ok=True)

    all_counts = pd.concat([counts_df, combined_counts_df], ignore_index=True)
    all_totals = pd.concat([totals_df, combined_totals_df], ignore_index=True)

    master_vol_lookup = volumes.set_index('order')['volume_mm3']
    meta_indexed = metadata.set_index('order')
    order_index = metadata['order'].values

    sample_keys = all_counts[['group', 'sample']].drop_duplicates().itertuples(index=False)
    for group, sample in sample_keys:
        sub = all_counts[(all_counts['group'] == group) & (all_counts['sample'] == sample)]
        totals = (all_totals[(all_totals['group'] == group) & (all_totals['sample'] == sample)]
                  .set_index('class_name')['total_valid'])

        # This sample's own warped-atlas volumes (falls back to the master-atlas
        # lookup if per_sample_volumes has no rows for this sample, e.g. an
        # unrecognized sample name).
        sample_vols = per_sample_volumes[per_sample_volumes['sample'] == sample]
        vol_lookup = sample_vols.set_index('order')['volume_mm3'] if not sample_vols.empty else master_vol_lookup

        pieces = []
        for cls, g in sub.groupby('class_name'):
            count = g.set_index('order')['count'].reindex(order_index).fillna(0.0)
            total_valid = totals.get(cls, 0)
            pct = (count / total_valid * 100.0) if total_valid > 0 else pd.Series(np.nan, index=count.index)
            vol = vol_lookup.reindex(count.index)
            with np.errstate(divide='ignore', invalid='ignore'):
                density = np.where(vol.values > 0, count.values / vol.values, np.nan)

            piece = meta_indexed.loc[count.index, ['id', 'graph_order', 'name', 'level']].copy()
            piece['order'] = piece.index
            piece['class_name'] = cls
            piece['formula'] = formula_info.get(cls, '')
            piece['count'] = count.values
            piece['percentage'] = pct.values
            piece['density'] = density
            piece['volume_mm3'] = vol.values
            pieces.append(piece.reset_index(drop=True))

        sample_table = pd.concat(pieces, ignore_index=True).sort_values(['level', 'class_name', 'order'])
        out_path = os.path.join(sample_dir, f'{sample}_region_registration_summary.xlsx')
        max_level = int(sample_table['level'].max()) if not sample_table.empty else -1
        with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
            for level in range(max_level + 1):
                sheet = sample_table[sample_table['level'] == level].sort_values(['class_name', 'order'])
                if sheet.empty:
                    continue
                sheet.to_excel(writer, sheet_name=f'L{level:02d}', index=False)
        print(f"Wrote per-sample region summary: {out_path} ({len(sample_table)} rows)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    args = parser.parse_args()

    cfg = load_config(args.config)
    (df, group_a_name, group_b_name, counts_df, totals_df, volumes, per_sample_volumes, metadata,
     combined_counts_df, combined_totals_df, formula_info) = run_all_stats(cfg)
    write_outputs(df, cfg, group_a_name, group_b_name)

    out_dir = cfg.get('output', {}).get('dir', './stats_output')
    write_per_sample_tables(counts_df, totals_df, volumes, per_sample_volumes, metadata,
                             combined_counts_df, combined_totals_df, formula_info, out_dir)


if __name__ == '__main__':
    main()
