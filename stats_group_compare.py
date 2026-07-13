# -*- coding: utf-8 -*-
"""
Group comparison statistics across the CCF ontology hierarchy.

Compares cell counts (Count / Percentage / Density) between two user-defined
groups of samples, for every cell class found under each sample's
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

BACKGROUND_NAMES = {'background', 'no label', ''}
METRICS = ('Count', 'Percentage', 'Density')


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
    x, y, z, xt, yt, zt, graph_order, name. Over-allocate column names since
    some rows can be ragged (trailing commas etc.)."""
    df = pd.read_csv(csv_path, header=None, names=range(20), engine='python')
    graph_order = pd.to_numeric(df[6], errors='coerce').fillna(-1).astype(int).values
    names = df[7].map(clean_name).values
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

def discover_classes(data_dir, sample_names, explicit=None):
    if explicit:
        return list(explicit)
    classes = set()
    for s in sample_names:
        reg_dir = os.path.join(data_dir, s, 'cell_registration')
        if os.path.isdir(reg_dir):
            classes.update(d for d in os.listdir(reg_dir) if os.path.isdir(os.path.join(reg_dir, d)))
    return sorted(classes)


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
                csv_path = os.path.join(data_dir, sample, 'cell_registration', cls, 'cell_registration.csv')
                bins, total = class_counts_for_sample(csv_path)
                count_frames.append(pd.DataFrame({
                    'group': group_key, 'sample': sample, 'class_name': cls,
                    'order': orders, 'count': bins,
                }))
                total_rows.append({'group': group_key, 'sample': sample, 'class_name': cls, 'total_valid': total})
    counts_df = pd.concat(count_frames, ignore_index=True)
    totals_df = pd.DataFrame(total_rows)
    return counts_df, totals_df


def metric_matrix(counts_df, totals_df, volumes_df, class_name, metric, group_key, samples):
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

    if metric == 'Density':
        vol = volumes_df.set_index('order')['volume_mm3'].reindex(pivot.index).values.astype(float)
        density = np.full_like(mat, np.nan)
        nonzero = vol > 0
        density[nonzero, :] = mat[nonzero, :] / vol[nonzero, None]
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


def run_level_tests(mat_a, mat_b, metadata, level, class_name, metric):
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
    return out


# ================= Orchestration =================

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

    classes = cfg.get('classes') or discover_classes(cfg['data_dir'], samples_a + samples_b)
    print(f"Classes: {classes}")
    print(f"Group A ({group_a_name}): {samples_a}")
    print(f"Group B ({group_b_name}): {samples_b}")

    counts_df, totals_df = collect_counts(cfg['data_dir'], classes, {'a': samples_a, 'b': samples_b})

    max_level = int(metadata['level'].max())
    all_rows = []
    for cls in classes:
        for metric in METRICS:
            mat_a = metric_matrix(counts_df, totals_df, volumes, cls, metric, 'a', samples_a)
            mat_b = metric_matrix(counts_df, totals_df, volumes, cls, metric, 'b', samples_b)
            for level in range(max_level + 1):
                res = run_level_tests(mat_a, mat_b, metadata, level, cls, metric)
                if not res.empty:
                    all_rows.append(res)

    result = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    if not result.empty:
        result[['fold_change', 'log2fc']] = result[['fold_change', 'log2fc']].replace([np.inf, -np.inf], np.nan)

    # attach region volumes for reference (joined by order, independent of class/metric loop above)
    result = result.merge(volumes[['order', 'volume_mm3']], on='order', how='left', suffixes=('', '_ref'))
    return result, group_a_name, group_b_name


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
            'column': ['level', 'order', 'id', 'graph_order', 'name', 'class_name', 'metric',
                       'n_a', 'n_b', 'mean_a', 'mean_b', 'fold_change', 'log2fc',
                       'mean_a_is_zero', 'p_value', 'p_fdr', 'volume_mm3'],
            'description': [
                'Ontology tree depth from root (0=whole brain, higher=finer subregions)',
                "ClearMap's internal dense region index (stable within one ontology file)",
                'Allen CCF structure id',
                "Allen ontology 'graph_order' value",
                'Region name',
                'Cell class / marker-combination name (from cell_registration/<class_name>/ folder)',
                'Count / Percentage / Density',
                f'Number of samples in group A ({group_a_name})',
                f'Number of samples in group B ({group_b_name})',
                f'Mean metric value, group A ({group_a_name})',
                f'Mean metric value, group B ({group_b_name})',
                'mean_b / mean_a (NaN if mean_a == 0 -- fold change undefined)',
                'log2(fold_change); NaN if undefined or infinite (complete presence/absence)',
                'True if mean_a == 0 (fold_change/log2fc undefined for this region)',
                "Welch's t-test p-value (unequal variance), raw/uncorrected",
                'Benjamini-Hochberg FDR-corrected p-value, computed separately within each level',
                'Region volume in mm^3 (from master atlas volume; used for Density metric)',
            ],
        })
        readme.to_excel(writer, sheet_name='ReadMe', index=False)

        volumes_out = df[['order', 'id', 'name', 'level', 'volume_mm3']].drop_duplicates().sort_values('order')
        volumes_out.to_excel(writer, sheet_name='Region_Volumes', index=False)

        max_level = int(df['level'].max()) if not df.empty else -1
        for level in range(max_level + 1):
            sheet = df[df['level'] == level].sort_values(['class_name', 'metric', 'p_fdr'])
            if sheet.empty:
                continue
            sheet.to_excel(writer, sheet_name=f'L{level:02d}', index=False)
    print(f"Wrote Excel workbook: {excel_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    args = parser.parse_args()

    cfg = load_config(args.config)
    df, group_a_name, group_b_name = run_all_stats(cfg)
    write_outputs(df, cfg, group_a_name, group_b_name)


if __name__ == '__main__':
    main()
