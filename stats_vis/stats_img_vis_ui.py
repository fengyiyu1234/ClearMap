import napari
import pandas as pd
import numpy as np
import tifffile
import SimpleITK as sitk
import os
import json
import re

# UI 相关 (新增了 QFileDialog 和 QMessageBox)
from PyQt5.QtWidgets import (QComboBox, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QFrame,
                             QCheckBox, QStackedWidget, QLineEdit, QPushButton, QDoubleSpinBox,
                             QFileDialog, QMessageBox, QGroupBox, QListWidget, QListWidgetItem,
                             QAbstractItemView)
from PyQt5.QtCore import Qt
# Matplotlib 相关
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from napari.utils.colormaps import Colormap

# ================= ⚙️ 用户配置区域 =================

CONFIG = {
    # 路径默认留空，由用户在 UI 中自行选择
    "parent_data_dir": "",
    # region_stats_by_level.xlsx produced by stats_group_compare.py: one
    # sheet per ontology level (L00, L01, ...) plus ReadMe/Region_Volumes
    # reference sheets. Pre-filled as default but still user-editable via UI.
    "stats_long_table": "/data/hdd12tb-1/fengyi/COMBINe/clearmap/TSC/stats/region_stats_by_level.xlsx",
    "std_atlas_path": "",
    "ontology_json_path": "",

    # 5. 分辨率参数 [X, Y, Z] (单位: um) -- 仅用于 Native 视图: 把 cell_centroids
    # 坐标 (config.yaml: resolution.original 分辨率) 缩放到 resample.tif 显示
    # 空间 (config.yaml: resolution.resampled 分辨率, 与 p5 atlas 分辨率一致)。
    # 与最终 atlas 文件本身无关 -- Atlas/Stats 模式下的坐标已经是 atlas 体素空间，不需要缩放。
    "res_original":  np.array([0.65, 0.65, 8.0]),
    "res_resampled": np.array([20.0, 20.0, 20.0]), # matches resample.tif / p5 atlas resolution

    # 6. cell_centroids 文件前缀 (native 模式下, ob_<class_name>.csv)
    "cell_prefix": "ob_",
}

# ================= 🧠 1. 脑区层级管理器 (基于 ClearMap.Alignment.Annotation) =================
class OntologyTree:
    """Self-contained Allen CCF ontology parser -- no ClearMap package import,
    so this interactive viewer can keep running in a lightweight napari/PyQt5
    environment (ClearMap itself pulls in heavy, environment-specific deps
    like graph_tool/elastix that this process shouldn't need). Tracks id,
    name, graph_order, tree level (depth from root) and parent id per node.

    Note: raw Allen atlas ids (.mhd/atlas volumes) and ClearMap graph_order
    values (cell_registration.csv's id column) are two different numbering
    systems that can collide numerically for different nodes -- use
    get_name() for the former and get_name_by_graph_order() for the latter.
    """
    def __init__(self, json_path=None):
        self.id_to_name = {}
        self.graph_order_to_name = {}
        self.name_to_id = {}
        self.id_to_level = {}
        self.id_to_parent = {}
        self.max_level = 0
        if json_path and os.path.exists(json_path):
            self._parse(json_path)

    def _parse(self, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        root = None
        if isinstance(data, dict):
            if 'msg' in data:
                msg = data['msg']
                root = msg[0] if isinstance(msg, list) else msg
            elif 'children' in data:
                root = data
        elif isinstance(data, list) and data:
            root = data[0]
        if root is None:
            return

        def walk(node, level, parent_id):
            node_id = node.get('id') or node.get('structure_id')
            graph_order = node.get('graph_order')
            name = node.get('name') or node.get('safe_name') or node.get('acronym')
            if name is not None:
                s_name = DataLoader.clean_part(name)
                if node_id is not None:
                    node_id = int(node_id)
                    self.id_to_name[node_id] = s_name
                    if s_name not in self.name_to_id:
                        self.name_to_id[s_name] = node_id
                    self.id_to_level[node_id] = level
                    self.id_to_parent[node_id] = parent_id
                if graph_order is not None:
                    self.graph_order_to_name[int(graph_order)] = s_name
                    if s_name not in self.name_to_id:
                        self.name_to_id[s_name] = int(graph_order)
            self.max_level = max(self.max_level, level)
            for child in (node.get('children') or []):
                walk(child, level + 1, node_id)

        walk(root, 0, None)

    def get_name(self, region_value):
        """Id-based lookup (used for raw atlas-volume ids: .mhd/.tif voxel
        values, hover/click). NOTE: a graph_order value can numerically
        collide with a *different* node's raw id -- use get_name_by_graph_order
        for values coming from cell_registration.csv's graph_order column."""
        try:
            return self.id_to_name.get(int(region_value), f"Region {region_value}")
        except (TypeError, ValueError):
            return f"Region {region_value}"

    def get_name_by_graph_order(self, graph_order_value):
        try:
            return self.graph_order_to_name.get(int(graph_order_value), f"Region {graph_order_value}")
        except (TypeError, ValueError):
            return f"Region {graph_order_value}"

    def level_of_id(self, region_id):
        return self.id_to_level.get(int(region_id))

    def ancestor_id_at_level(self, region_id, level):
        """Walk up the parent chain until at or above `level` (a shallower
        node just maps to itself -- mirrors how a chosen level's stats are
        painted onto a deeper-resolution atlas)."""
        rid = int(region_id)
        while self.id_to_level.get(rid, 0) > level:
            parent = self.id_to_parent.get(rid)
            if parent is None: break
            rid = parent
        return rid

    def ancestor_ids_at_level(self, ids_array, level):
        return np.array([self.ancestor_id_at_level(i, level) for i in ids_array], dtype=int)

# ================= 📂 2. 数据加载与处理 =================
class DataLoader:
    @staticmethod
    def clean_part(val):
        if pd.isna(val): return ""
        s = str(val).strip()
        s = re.sub(r"^b['\"]", "", s)
        s = re.sub(r"['\"]$", "", s)
        return s.strip().strip(',')

    @staticmethod
    def scan_samples(parent_dir):
        samples = {}
        if not parent_dir or not os.path.exists(parent_dir): return samples
        for entry in os.scandir(parent_dir):
            if not entry.is_dir(): continue
            name = entry.name

            res_path = os.path.join(entry.path, 'resampled.tif')
            mhd_path = os.path.join(entry.path, 'volume', 'result.mhd')

            if os.path.exists(res_path):
                samples[name] = {
                    'path': entry.path,
                    'resampled': res_path, 'mhd': mhd_path,
                    'cell_dir_raw': os.path.join(entry.path, 'cell_centroids'),
                    'cell_dir_reg': os.path.join(entry.path, 'cell_registration')
                }
        return samples

    @staticmethod
    def normalize_class_key(name):
        """Token-set key for fuzzy class matching: lowercase, split on
        non-alphanumerics, drop pure-numeric tokens (batch/version markers
        like the '3' in 'glia_3_GFP') so naming drift across samples -- e.g.
        'glia_3_GFP' vs 'glia_GFP' -- still resolves to the same class."""
        tokens = re.split(r'[^a-zA-Z0-9]+', name)
        return frozenset(t.lower() for t in tokens if t and not t.isdigit())

    @staticmethod
    def discover_classes(samples):
        """Union of cell_registration/<class_name>/ subfolder names across all
        scanned samples -- cell classes are now arbitrary marker-combination
        names (e.g. 'glia_3_GFP'), not a fixed enum. Names are grouped by
        normalize_class_key so differently-named variants of the same marker
        combination on different samples (e.g. 'glia_3_GFP' vs 'glia_GFP')
        collapse into a single class instead of appearing as two."""
        all_names = set()
        for info in samples.values():
            reg_dir = info.get('cell_dir_reg')
            if reg_dir and os.path.isdir(reg_dir):
                all_names.update(d for d in os.listdir(reg_dir) if os.path.isdir(os.path.join(reg_dir, d)))

        groups = {}
        for name in sorted(all_names):
            groups.setdefault(DataLoader.normalize_class_key(name), []).append(name)
        canonical = [min(names, key=lambda n: (len(n), n)) for names in groups.values()]
        return sorted(canonical)

    @staticmethod
    def resolve_class_dir(reg_dir, class_name):
        """Find the actual cell_registration/<...> subfolder for `class_name`
        within one sample, tolerating naming drift (see normalize_class_key).
        Returns None if no folder matches."""
        if not os.path.isdir(reg_dir):
            return None
        actual = [d for d in os.listdir(reg_dir) if os.path.isdir(os.path.join(reg_dir, d))]
        if class_name in actual:
            return class_name
        key = DataLoader.normalize_class_key(class_name)
        matches = [d for d in actual if DataLoader.normalize_class_key(d) == key]
        return matches[0] if matches else None

    @staticmethod
    def resolve_class_file(folder_path, class_name, prefix, suffix='.csv'):
        """Find the actual '<prefix><...><suffix>' file for `class_name`
        within one sample's cell_centroids folder (native mode uses flat
        files, not subfolders), tolerating naming drift. Returns None if no
        file matches."""
        if not os.path.isdir(folder_path):
            return None
        exact = f"{prefix}{class_name}{suffix}"
        if os.path.exists(os.path.join(folder_path, exact)):
            return exact
        key = DataLoader.normalize_class_key(class_name)
        matches = []
        for fname in os.listdir(folder_path):
            if fname.startswith(prefix) and fname.endswith(suffix):
                stem = fname[len(prefix):-len(suffix)] if suffix else fname[len(prefix):]
                if DataLoader.normalize_class_key(stem) == key:
                    matches.append(fname)
        return matches[0] if matches else None

    @staticmethod
    def _read_centroids_csv(csv_path):
        """Read a cell_centroids/ob_<class>.csv file, tolerating both the
        newer run_inference.py header format (cx,cy,z,score,slice_name,
        tile_name -- see brain_detector/scripts/run_inference.py) and older
        header-less x,y,z-only files. Always returns a DataFrame with at
        least cx/cy/z columns; missing score/slice_name/tile_name are filled
        with NaN so callers don't need to special-case legacy files."""
        expected = ['cx', 'cy', 'z', 'score', 'slice_name', 'tile_name']
        df = pd.read_csv(csv_path)
        cols = [str(c).strip().lower() for c in df.columns]
        if cols[:3] not in (['cx', 'cy', 'z'], ['x', 'y', 'z']):
            # First row wasn't actually a header -- re-read raw and name
            # columns positionally.
            df = pd.read_csv(csv_path, header=None)
            df.columns = expected[:df.shape[1]]
        else:
            df.columns = cols
            if df.columns[0] == 'x':
                df = df.rename(columns={'x': 'cx', 'y': 'cy'})
        for col in ('score', 'slice_name', 'tile_name'):
            if col not in df.columns:
                df[col] = np.nan
        return df

    @staticmethod
    def load_mhd(path):
        if not os.path.exists(path): return None
        return sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.uint32)

    @staticmethod
    def normalize_image_8bit(img_path):
        if not os.path.exists(img_path): return None, None
        img = tifffile.imread(img_path)
        low, high = np.percentile(img, [0.5, 95.5])
        img_clipped = np.clip(img, low, high)
        return ((img_clipped - low) / (high - low) * 255).astype(np.uint8), img.shape

    @staticmethod
    def load_cells_native_df(folder_path, raw_res, target_res, mhd_data, ontology, class_names, sample_name=None):
        all_dfs = []
        scale_factor = raw_res / target_res
        mhd_shape = mhd_data.shape if mhd_data is not None else (0,0,0)

        for class_name in class_names:
            fname = DataLoader.resolve_class_file(folder_path, class_name, CONFIG['cell_prefix'])
            csv_path = os.path.join(folder_path, fname) if fname else None
            if csv_path and os.path.exists(csv_path):
                df_src = DataLoader._read_centroids_csv(csv_path)
                if len(df_src) == 0: continue
                raw_xyz = df_src[['cx', 'cy', 'z']].values.astype(float)
                napari_pts = (raw_xyz * scale_factor)[:, [2, 1, 0]]

                ids = []
                for p in napari_pts:
                    z, y, x = int(round(p[0])), int(round(p[1])), int(round(p[2]))
                    if mhd_data is not None and 0 <= z < mhd_shape[0] and 0 <= y < mhd_shape[1] and 0 <= x < mhd_shape[2]:
                        ids.append(mhd_data[z, y, x])
                    else:
                        ids.append(0)

                df_clean = pd.DataFrame(napari_pts, columns=['z', 'y', 'x'])
                df_clean['class_name'] = class_name
                df_clean['mapped_id'] = ids
                df_clean['region'] = [ontology.get_name(uid) for uid in ids]
                df_clean['sample'] = sample_name
                # Provenance back to the raw TB-scale tile/slice this detection
                # came from (see brain_detector/scripts/run_inference.py) --
                # lets the user pull up the exact source tile for a cell they
                # flag as suspicious in the viewer.
                df_clean['raw_x'] = raw_xyz[:, 0]
                df_clean['raw_y'] = raw_xyz[:, 1]
                df_clean['raw_z'] = raw_xyz[:, 2]
                df_clean['score'] = df_src['score'].values
                df_clean['slice_name'] = df_src['slice_name'].values
                df_clean['tile_name'] = df_src['tile_name'].values
                all_dfs.append(df_clean)
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    @staticmethod
    def load_cells_atlas_df(folder_path, ontology, class_names, sample_name=None, raw_centroids_dir=None):
        all_dfs = []
        for class_name in class_names:
            actual_dir = DataLoader.resolve_class_dir(folder_path, class_name)
            csv_path = os.path.join(folder_path, actual_dir, 'cell_registration.csv') if actual_dir else None
            if csv_path and os.path.exists(csv_path):
                try:
                    # cell_registration.csv columns (no header): x,y,z (raw
                    # pixel-space centroid, cellMap.py's untransformed input),
                    # xr,yr,zr (resampled-space, pre-elastix), xt,yt,zt
                    # (atlas-space, post-elastix), graph_order, name, and --
                    # for files written by the current cellMap.py --
                    # slice_name (11), tile_name (12), score (13).
                    df_raw = pd.read_csv(csv_path, header=None, names=range(20), engine='python')
                    if len(df_raw) > 0:
                        coords = df_raw.iloc[:, 6:9].values.astype(float)
                        valid_mask = ~np.isnan(coords).any(axis=1)
                        row_idx = np.where(valid_mask)[0]
                        coords = coords[valid_mask]

                        ids = df_raw.iloc[:, 9].values[valid_mask]  # graph_order, not raw atlas id
                        ids = pd.to_numeric(pd.Series(ids), errors='coerce').fillna(0).astype(int).values

                        napari_pts = coords[:, [2, 1, 0]]

                        df_clean = pd.DataFrame(napari_pts, columns=['z', 'y', 'x'])
                        df_clean['class_name'] = class_name
                        df_clean['mapped_id'] = ids
                        df_clean['region'] = [ontology.get_name_by_graph_order(uid) for uid in ids]
                        df_clean['sample'] = sample_name

                        # x,y,z (cols 0-2) are always cellMap.py's raw
                        # pixel-space input coordinates -- no join needed to
                        # recover these.
                        raw_xyz = df_raw.iloc[:, 0:3].values.astype(float)[valid_mask]
                        df_clean['raw_x'] = raw_xyz[:, 0]
                        df_clean['raw_y'] = raw_xyz[:, 1]
                        df_clean['raw_z'] = raw_xyz[:, 2]

                        # Tile/slice/score: prefer the columns the current
                        # cellMap.py embeds directly (no cell_centroids/
                        # dependency). Older registration files predate this
                        # and fall back to a row-index join -- cellMap.py
                        # processes points strictly in-order without
                        # reordering or dropping rows, so row i here == row i
                        # in the matching cell_centroids/ob_<class>.csv.
                        has_embedded = df_raw.shape[1] > 12 and df_raw.iloc[:, 12].notna().any()
                        if has_embedded:
                            df_clean['slice_name'] = df_raw.iloc[:, 11].values[valid_mask]
                            df_clean['tile_name'] = df_raw.iloc[:, 12].values[valid_mask]
                            df_clean['score'] = (pd.to_numeric(df_raw.iloc[:, 13], errors='coerce').values[valid_mask]
                                                  if df_raw.shape[1] > 13 else np.nan)
                        else:
                            df_clean['slice_name'] = np.nan
                            df_clean['tile_name'] = np.nan
                            df_clean['score'] = np.nan
                            fname = (DataLoader.resolve_class_file(raw_centroids_dir, class_name, CONFIG['cell_prefix'])
                                      if raw_centroids_dir else None)
                            if fname:
                                try:
                                    df_src = DataLoader._read_centroids_csv(os.path.join(raw_centroids_dir, fname))
                                    if len(df_src) == len(df_raw):
                                        joined = df_src.iloc[row_idx].reset_index(drop=True)
                                        df_clean['score'] = joined['score'].values
                                        df_clean['slice_name'] = joined['slice_name'].values
                                        df_clean['tile_name'] = joined['tile_name'].values
                                    else:
                                        print(f"⚠️ '{class_name}' 行数不匹配 (registration={len(df_raw)} vs "
                                              f"centroids={len(df_src)})，跳过 tile/slice 关联。")
                                except Exception as e:
                                    print(f"⚠️ '{class_name}' tile/slice 关联失败: {e}")

                        all_dfs.append(df_clean)
                except Exception as e:
                    print(f"❌ 解析 '{class_name}' 坐标出错: {e}")
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    @staticmethod
    def load_stats(long_table_path):
        """Loads region-level stats produced by stats_group_compare.py.
        Supports the tidy long-format CSV (region_stats.csv) as well as the
        by-level workbook (region_stats_by_level.xlsx), which splits the same
        long table across one sheet per ontology level (L00, L01, ...) plus
        'ReadMe'/'Region_Volumes' reference sheets that hold no stats rows and
        are skipped. Columns include: level, id, name, class_name, metric,
        fold_change, log2fc, p_value, p_fdr, ... Returns the concatenated
        DataFrame -- filtering by class/metric/level happens in
        refresh_heatmaps()."""
        if not long_table_path or not os.path.exists(long_table_path):
            return pd.DataFrame()
        try:
            if long_table_path.lower().endswith(('.xlsx', '.xls')):
                xls = pd.ExcelFile(long_table_path)
                level_sheets = [s for s in xls.sheet_names if re.fullmatch(r'L\d+', s)]
                dfs = [xls.parse(s) for s in level_sheets]
                return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            return pd.read_csv(long_table_path)
        except Exception as e:
            print(f"❌ Stats table error: {e}")
            return pd.DataFrame()

# ================= 🎮 3. 主控制器 =================
class MainController:
    def __init__(self, viewer):
        self.viewer = viewer
        
        # 初始状态为空，等待用户点击加载
        self.ontology = None
        self.samples = {}
        self.classes = []
        self.all_stats_df = pd.DataFrame()

        self.current_atlas_labels = None
        self.current_cells_df = pd.DataFrame()
        self.highlight_atlas = None
        self.highlight_cells = None
        self.last_hover_val = -1
        self._atlas_unique_ids = None
        self._atlas_inverse = None

        self.mode = "Stats"
        self.current_class = None
        self.current_metric = "Count"
        self.current_level = 0
        self.cell_checkboxes = {}
        self.last_search_mode = "Exact"

        # 🚩 Flagging: click a cell -> pin it -> export tile/slice provenance
        # for suspicious cells so the user can pull up the raw TB-scale tile.
        self.flagged_cells = []       # list[dict], persists across samples/views
        self.last_clicked_cell = None # dict, most recent cell click (candidate to pin)
        self.flag_layer = None        # current napari points layer showing pins for this view
        self.last_highlighted_df = pd.DataFrame()  # last Search Regions result, for bulk export

        self.setup_ui()
        self.setup_callbacks()

    def get_dark_colormap(self):
        return Colormap(np.array([[0,0,1,1], [0,0,0,0], [1,0,0,1]]), name='BBR', interpolation='linear')

    def setup_highlight_layers(self, shape):
        for name in [">> Highlight Atlas <<", ">> Highlight Cells <<", "✨ Selection"]:
            if name in self.viewer.layers: self.viewer.layers.remove(name)
                
        self.highlight_atlas = self.viewer.add_labels(np.zeros(shape, dtype=np.uint32), name=">> Highlight Atlas <<", opacity=0.8)
        self.highlight_cells = self.viewer.add_points(np.empty((0, 3)), ndim=3, name=">> Highlight Cells <<",
                                                      face_color='white', border_color='yellow', size=self.spin_point_size.value(), opacity=1.0)

    def render_cells_from_df(self, df_cells, labels_layer):
        if df_cells.empty or labels_layer is None: return

        id_color_map = {0: np.array([0.5, 0.5, 0.5, 1.0])}
        for uid in df_cells['mapped_id'].unique():
            if uid != 0: id_color_map[uid] = labels_layer.get_color(uid)

        for cls_name in self.classes:
            sub_df = df_cells[df_cells['class_name'] == cls_name]
            if len(sub_df) > 0:
                coords = sub_df[['z', 'y', 'x']].values
                colors = np.array([id_color_map[uid] for uid in sub_df['mapped_id']])
                is_vis = self.cell_checkboxes[cls_name].isChecked()

                # One layer per discovered cell class -- napari's default
                # per-layer face-color cycling distinguishes layers, no
                # per-class symbol/shape mapping needed.
                layer = self.viewer.add_points(
                    coords, name=f"Cell: {cls_name}", face_color=colors,
                    symbol='disc', size=self.spin_point_size.value(), border_width=0, blending='translucent', visible=is_vis
                )
                # Carried through for click-to-pin export (tile/slice
                # provenance -- see DataLoader.load_cells_native_df /
                # load_cells_atlas_df).
                layer.features = pd.DataFrame({
                    'Region': sub_df['region'].values,
                    'Class': sub_df['class_name'].values,
                    'Sample': sub_df['sample'].values,
                    'Tile': sub_df['tile_name'].values,
                    'Slice': sub_df['slice_name'].values,
                    'Score': sub_df['score'].values,
                    'RawX': sub_df['raw_x'].values,
                    'RawY': sub_df['raw_y'].values,
                    'RawZ': sub_df['raw_z'].values,
                })
                layer.events.highlight.connect(self.on_cell_layer_click)

    def perform_search(self, search_mode=None):
        if search_mode is not None:
            self.last_search_mode = search_mode
        search_mode = self.last_search_mode

        keyword = self.input_search.text().strip()
        self.last_highlighted_df = pd.DataFrame()

        if not keyword:
            if self.highlight_cells: self.highlight_cells.data = np.empty((0, 3))
            if self.highlight_atlas and self.current_atlas_labels is not None:
                self.highlight_atlas.data = np.zeros_like(self.current_atlas_labels)
            self.viewer.status = "Ready."
            return
            
        self.viewer.status = f"Searching: {keyword}..."
        
        matched_ids = []
        if self.ontology:
            if search_mode == 'Exact':
                for name, region_id in self.ontology.name_to_id.items():
                    if name.lower() == keyword.lower(): matched_ids.append(region_id)
            else:
                for name, region_id in self.ontology.name_to_id.items():
                    if keyword.lower() in name.lower(): matched_ids.append(region_id)
                
        if matched_ids and self.current_atlas_labels is not None:
            mask = np.isin(self.current_atlas_labels, matched_ids)
            h_data = np.zeros_like(self.current_atlas_labels)
            h_data[mask] = self.current_atlas_labels[mask]
            self.highlight_atlas.data = h_data
        else:
            if self.highlight_atlas: self.highlight_atlas.data = np.zeros_like(self.current_atlas_labels)
            
        if not self.current_cells_df.empty:
            active_classes = [name for name, cb in self.cell_checkboxes.items() if cb.isChecked()]
            if search_mode == 'Exact':
                region_mask = self.current_cells_df['region'].str.lower() == keyword.lower()
            else:
                region_mask = self.current_cells_df['region'].str.contains(keyword, case=False, regex=False)
                
            class_mask = self.current_cells_df['class_name'].isin(active_classes)
            subset_df = self.current_cells_df[region_mask & class_mask]
            subset_points = subset_df[['z', 'y', 'x']].values

            self.highlight_cells.data = subset_points if len(subset_points) > 0 else np.empty((0, 3))
            self.last_highlighted_df = subset_df.copy()
            self.viewer.status = f"✅ [{search_mode}] Found {len(matched_ids)} regions | Cells: {len(subset_points)}"

    def load_standard_view(self):
        self.viewer.layers.clear()
        self.current_cells_df = pd.DataFrame()
        atlas_path = CONFIG['std_atlas_path']
        
        if atlas_path and os.path.exists(atlas_path):
            data = tifffile.imread(atlas_path) if atlas_path.lower().endswith(('.tif', '.tiff')) else __import__('nrrd').read(atlas_path)[0]
            self.current_atlas_labels = data.astype(np.uint32)
            self._atlas_unique_ids, self._atlas_inverse = None, None
            self.viewer.add_labels(self.current_atlas_labels, name="Atlas Anatomy", opacity=0.1)
            self.setup_highlight_layers(self.current_atlas_labels.shape)
            self.refresh_heatmaps()

    def load_sample_native_view(self, sample_key):
        self.viewer.layers.clear()
        self.current_cells_df = pd.DataFrame()
        s = self.samples[sample_key]
        
        img_norm, shape = DataLoader.normalize_image_8bit(s['resampled'])
        if img_norm is not None:
            self.viewer.add_image(img_norm, name="Raw Image", colormap="gray", blending='additive')

        mhd = DataLoader.load_mhd(s['mhd'])
        labels_layer = None
        if mhd is not None:
            self.current_atlas_labels = mhd
            self._atlas_unique_ids, self._atlas_inverse = None, None
            labels_layer = self.viewer.add_labels(mhd, name="Atlas Regions", opacity=0.05, visible=False)
            self.setup_highlight_layers(mhd.shape)

        df_cells = DataLoader.load_cells_native_df(s['cell_dir_raw'], CONFIG['res_original'], CONFIG['res_resampled'], mhd, self.ontology, self.classes, sample_key)
        self.current_cells_df = df_cells
        self.render_cells_from_df(df_cells, labels_layer)
        self.setup_flag_layer(sample_key)

    def load_sample_atlas_view(self, sample_key):
        self.viewer.layers.clear()
        self.current_cells_df = pd.DataFrame()
        s = self.samples[sample_key]

        atlas_layer = None
        atlas_path = CONFIG['std_atlas_path']
        
        if atlas_path and os.path.exists(atlas_path):
            data = tifffile.imread(atlas_path) if atlas_path.lower().endswith(('.tif', '.tiff')) else __import__('nrrd').read(atlas_path)[0]
            self.current_atlas_labels = data.astype(np.uint32)
            self._atlas_unique_ids, self._atlas_inverse = None, None
            atlas_layer = self.viewer.add_labels(self.current_atlas_labels, name="Atlas Anatomy", opacity=0.3)
            self.setup_highlight_layers(self.current_atlas_labels.shape)
        
        df_cells = DataLoader.load_cells_atlas_df(s['cell_dir_reg'], self.ontology, self.classes, sample_key, s['cell_dir_raw'])
        self.current_cells_df = df_cells
        self.render_cells_from_df(df_cells, atlas_layer)
        self.setup_flag_layer(sample_key)

    def on_cell_layer_click(self, event):
        layer = event.source
        if self.viewer.layers.selection.active != layer: return
        if len(layer.selected_data) > 0:
            idx = list(layer.selected_data)[0]
            full_name = layer.features['Region'].iloc[idx]
            self.input_search.setText(full_name)
            self.perform_search("Exact")

            pt = layer.data[idx]
            self.last_clicked_cell = {
                'sample': layer.features['Sample'].iloc[idx],
                'mode': self.mode,
                'class_name': layer.features['Class'].iloc[idx],
                'region': full_name,
                'z': float(pt[0]), 'y': float(pt[1]), 'x': float(pt[2]),
                'raw_x': layer.features['RawX'].iloc[idx],
                'raw_y': layer.features['RawY'].iloc[idx],
                'raw_z': layer.features['RawZ'].iloc[idx],
                'score': layer.features['Score'].iloc[idx],
                'slice_name': layer.features['Slice'].iloc[idx],
                'tile_name': layer.features['Tile'].iloc[idx],
            }
            self.lbl_last_click.setText(self._format_last_click(self.last_clicked_cell))

            layer.selected_data = set()

    # --- 🚩 Flagging: pin suspicious cells, export tile/slice provenance ---
    @staticmethod
    def _fmt(v, nd=1):
        try:
            if v is None or (isinstance(v, float) and np.isnan(v)): return '—'
            if isinstance(v, (int, float, np.floating, np.integer)): return f"{float(v):.{nd}f}"
            return str(v)
        except Exception:
            return '—'

    def _format_last_click(self, c):
        return (f"Region: {c['region']}  |  Class: {c['class_name']}\n"
                f"Tile: {self._fmt(c.get('tile_name'), 0)}   Slice: {self._fmt(c.get('slice_name'), 0)}\n"
                f"Raw XYZ: ({self._fmt(c.get('raw_x'))}, {self._fmt(c.get('raw_y'))}, {self._fmt(c.get('raw_z'))})"
                f"   Score: {self._fmt(c.get('score'), 3)}")

    def setup_flag_layer(self, sample_key):
        """(Re)create the '🚩 Flagged Cells' points layer for the current
        sample+mode, restoring any previously-pinned points so flags survive
        switching between samples/views (self.flagged_cells is the durable
        store; this layer is just its visual projection onto the current
        view's coordinate space)."""
        if "🚩 Flagged Cells" in self.viewer.layers:
            self.viewer.layers.remove("🚩 Flagged Cells")
        matches = [f for f in self.flagged_cells if f['sample'] == sample_key and f['mode'] == self.mode]
        pts = np.array([[f['z'], f['y'], f['x']] for f in matches]) if matches else np.empty((0, 3))
        self.flag_layer = self.viewer.add_points(
            pts, ndim=3, name="🚩 Flagged Cells", symbol='star',
            face_color='red', border_color='yellow', border_width=0.1,
            size=self.spin_point_size.value() * 2.2, opacity=0.95
        )

    def pin_last_clicked_cell(self):
        if not self.last_clicked_cell:
            QMessageBox.information(None, "Info", "还没有点击过任何细胞。请先在视图中点击一个细胞点，再点 Pin。")
            return
        entry = dict(self.last_clicked_cell)
        entry['pinned_at'] = pd.Timestamp.now().isoformat(timespec='seconds')
        self.flagged_cells.append(entry)
        if self.flag_layer is not None:
            pt = np.array([[entry['z'], entry['y'], entry['x']]])
            self.flag_layer.data = np.vstack([self.flag_layer.data, pt]) if len(self.flag_layer.data) else pt
        self.lbl_flag_count.setText(f"🚩 已标记 {len(self.flagged_cells)} 个细胞")

    def export_flagged_cells(self):
        if not self.flagged_cells:
            QMessageBox.information(None, "Info", "还没有标记任何细胞。")
            return
        path, _ = QFileDialog.getSaveFileName(None, "Export Flagged Cells", "flagged_cells.csv", "CSV Files (*.csv)")
        if not path: return
        cols = ['sample', 'mode', 'class_name', 'region', 'tile_name', 'slice_name',
                'raw_x', 'raw_y', 'raw_z', 'z', 'y', 'x', 'score', 'pinned_at']
        pd.DataFrame(self.flagged_cells)[cols].to_csv(path, index=False)
        QMessageBox.information(None, "Exported", f"已导出 {len(self.flagged_cells)} 个标记细胞 → {path}")

    def clear_flagged_cells(self):
        if not self.flagged_cells: return
        reply = QMessageBox.question(None, "Confirm", f"确定清空全部 {len(self.flagged_cells)} 个标记吗？此操作不可撤销。",
                                      QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes: return
        self.flagged_cells = []
        if self.flag_layer is not None:
            self.flag_layer.data = np.empty((0, 3))
        self.lbl_flag_count.setText("🚩 已标记 0 个细胞")

    def export_highlighted_cells(self):
        """Bulk export every cell currently matched by Search Regions --
        for when an entire anomalous region/cluster needs its source
        tiles/slices pulled, not just one outlier cell."""
        df = self.last_highlighted_df
        if df is None or df.empty:
            QMessageBox.information(None, "Info", "当前没有高亮的细胞可导出。请先用 Search Regions 搜索一个脑区。")
            return
        path, _ = QFileDialog.getSaveFileName(None, "Export Highlighted Cells", "highlighted_cells.csv", "CSV Files (*.csv)")
        if not path: return
        cols = ['sample', 'class_name', 'region', 'mapped_id', 'tile_name', 'slice_name',
                'raw_x', 'raw_y', 'raw_z', 'z', 'y', 'x', 'score']
        cols = [c for c in cols if c in df.columns]
        df[cols].to_csv(path, index=False)
        QMessageBox.information(None, "Exported", f"已导出 {len(df)} 个细胞 → {path}")

    def refresh_heatmaps(self):
        if self.mode != "Stats" or self.current_atlas_labels is None: return
        for layer in list(self.viewer.layers):
            if "Stats:" in layer.name: self.viewer.layers.remove(layer)

        if self.all_stats_df.empty or not self.current_class:
            return

        sub = self.all_stats_df[
            (self.all_stats_df['class_name'] == self.current_class) &
            (self.all_stats_df['metric'] == self.current_metric) &
            (self.all_stats_df['level'] == self.current_level)
        ]
        if sub.empty:
            self.viewer.status = f"No stats for {self.current_class} / {self.current_metric} / level {self.current_level}"
            return

        # Atlas is always stored at full (leaf) resolution; a chosen level's
        # region ids are ancestors of those leaf ids. Decompose the volume into
        # its (small) set of unique ids once, then only recompute the
        # level-ancestor lookup (cheap) when class/metric/level change.
        if self._atlas_unique_ids is None:
            self._atlas_unique_ids, self._atlas_inverse = np.unique(self.current_atlas_labels, return_inverse=True)
        ancestor_ids = self.ontology.ancestor_ids_at_level(self._atlas_unique_ids, self.current_level)

        log2fc_by_id = dict(zip(sub['id'], sub['log2fc'].fillna(0.0)))
        p_by_id = dict(zip(sub['id'], sub['p_value']))
        p_fdr_by_id = dict(zip(sub['id'], sub['p_fdr']))

        vals_raw = np.zeros(len(self._atlas_unique_ids))
        vals_fdr = np.zeros(len(self._atlas_unique_ids))
        for i, aid in enumerate(ancestor_ids):
            aid = int(aid)
            val = log2fc_by_id.get(aid)
            if val is None: continue
            if p_by_id.get(aid, 1.0) <= 0.05: vals_raw[i] = val
            if p_fdr_by_id.get(aid, 1.0) <= 0.05: vals_fdr[i] = val

        lut_raw_image = vals_raw[self._atlas_inverse].reshape(self.current_atlas_labels.shape)
        lut_fdr_image = vals_fdr[self._atlas_inverse].reshape(self.current_atlas_labels.shape)

        dark_cmap = self.get_dark_colormap()
        level_tag = f"L{self.current_level:02d}"
        self.viewer.add_image(lut_raw_image, name=f"Stats: {self.current_metric} {level_tag} (Raw P)", colormap=dark_cmap, contrast_limits=[-2,2], blending='additive', visible=True)
        self.viewer.add_image(lut_fdr_image, name=f"Stats: {self.current_metric} {level_tag} (FDR)", colormap=dark_cmap, contrast_limits=[-2,2], blending='additive', visible=False)

    def setup_callbacks(self):
        @self.viewer.mouse_move_callbacks.append
        def on_mouse_move(viewer, event):
            if self.current_atlas_labels is None or not self.ontology: return
            cursor = viewer.cursor.position
            if len(cursor) == 3:
                z, y, x = int(round(cursor[0])), int(round(cursor[1])), int(round(cursor[2]))
                shape = self.current_atlas_labels.shape
                if 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]:
                    val = self.current_atlas_labels[z, y, x]
                    if val != self.last_hover_val:
                        self.last_hover_val = val
                        if val > 0:
                            region_name = self.ontology.get_name(val)
                            self.lbl_hover.setText(f"📍 Hover: {region_name} (ID: {val})")
                            viewer.status = f"🧠 {region_name} (ID: {val})"
                        else:
                            self.lbl_hover.setText("📍 Hover: Background")
                            viewer.status = ""

        @self.viewer.mouse_drag_callbacks.append
        def on_click(viewer, event):
            active_layer = viewer.layers.selection.active
            if event.type != 'mouse_press' or self.current_atlas_labels is None or not self.ontology: return
            if active_layer is not None and active_layer.mode == 'pan_zoom': return
            
            c = np.round(viewer.cursor.position).astype(int)
            shape = self.current_atlas_labels.shape
            if not all(0 <= c[i] < shape[i] for i in range(3)): return
            
            rid = self.current_atlas_labels[c[0], c[1], c[2]]
            if rid > 0: 
                name = self.ontology.get_name(rid)
                if name and not name.startswith("Region"):
                    self.input_search.setText(name)
                    self.perform_search("Exact")

    def setup_ui(self):
        dock = QWidget()
        dock.setMaximumWidth(340) 
        layout = QVBoxLayout(dock)
        
        # --- 新增：0. 数据导入区域 ---
        group_data = QGroupBox("📁 0. Data Import")
        layout_data = QVBoxLayout(group_data)
        
        # Helper 函数创建带按钮的行
        def create_path_row(label, initial_text=""):
            h = QHBoxLayout()
            h.addWidget(QLabel(label))
            btn = QPushButton("Browse")
            line = QLineEdit()
            line.setText(initial_text)
            line.setReadOnly(True)
            h.addWidget(line)
            h.addWidget(btn)
            return h, line, btn

        r1, self.line_dir, btn_dir = create_path_row("Samples Dir:")
        r2, self.line_stats, btn_stats = create_path_row("Stats Table (xlsx/csv):", CONFIG['stats_long_table'])
        r3, self.line_atlas, btn_atlas = create_path_row("Atlas (.tif):")
        r4, self.line_json, btn_json = create_path_row("Ontology JSON:")

        btn_dir.clicked.connect(lambda: self.line_dir.setText(QFileDialog.getExistingDirectory(dock, "Select Samples Directory")))
        btn_stats.clicked.connect(lambda: self.line_stats.setText(QFileDialog.getOpenFileName(dock, "Select Stats Table", "", "Excel/CSV Files (*.xlsx *.xls *.csv)")[0]))
        btn_atlas.clicked.connect(lambda: self.line_atlas.setText(QFileDialog.getOpenFileName(dock, "Select Atlas", "", "Image Files (*.tif *.nrrd)")[0]))
        btn_json.clicked.connect(lambda: self.line_json.setText(QFileDialog.getOpenFileName(dock, "Select Ontology", "", "JSON Files (*.json)")[0]))

        layout_data.addLayout(r1); layout_data.addLayout(r2)
        layout_data.addLayout(r3); layout_data.addLayout(r4)

        self.btn_load = QPushButton("🚀 Load / Refresh Data")
        self.btn_load.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px;")
        self.btn_load.clicked.connect(self.process_loaded_data)
        layout_data.addWidget(self.btn_load)
        
        layout.addWidget(group_data)
        layout.addSpacing(5); line0 = QFrame(); line0.setFrameShape(QFrame.HLine); layout.addWidget(line0); layout.addSpacing(5)

        # --- 原有 UI ---
        layout.addWidget(QLabel("<b>1. View Mode:</b>"))
        self.combo_sample = QComboBox()
        self.combo_sample.addItem("等待加载数据...")
        self.combo_sample.currentTextChanged.connect(self.on_mode_change)
        layout.addWidget(self.combo_sample)

        h_size = QHBoxLayout()
        h_size.addWidget(QLabel("<b>Cell Size:</b>"))
        self.spin_point_size = QDoubleSpinBox()
        self.spin_point_size.setRange(0.1, 50.0); self.spin_point_size.setValue(5.0); self.spin_point_size.setSingleStep(1.0)
        self.spin_point_size.valueChanged.connect(self.on_point_size_change)
        h_size.addWidget(self.spin_point_size)
        layout.addLayout(h_size)

        layout.addSpacing(5); line1 = QFrame(); line1.setFrameShape(QFrame.HLine); layout.addWidget(line1); layout.addSpacing(5)

        layout.addWidget(QLabel("<b>🔍 Search Regions:</b>"))
        self.input_search = QLineEdit(); self.input_search.setPlaceholderText("Region name...")
        self.input_search.returnPressed.connect(lambda: self.perform_search())
        layout.addWidget(self.input_search)
        
        h_search_btns = QHBoxLayout()
        btn_fuzzy = QPushButton("Fuzzy Search"); btn_exact = QPushButton("Exact Search")
        btn_fuzzy.clicked.connect(lambda: self.perform_search("Fuzzy"))
        btn_exact.clicked.connect(lambda: self.perform_search("Exact"))
        h_search_btns.addWidget(btn_fuzzy); h_search_btns.addWidget(btn_exact)
        layout.addLayout(h_search_btns)

        btn_export_highlight = QPushButton("📤 Export Highlighted Cells (tile/slice)")
        btn_export_highlight.clicked.connect(self.export_highlighted_cells)
        layout.addWidget(btn_export_highlight)

        self.lbl_hover = QLabel("📍 Hover: None")
        self.lbl_hover.setStyleSheet("color: #888; font-size: 11px;")
        self.lbl_hover.setWordWrap(True); self.lbl_hover.setFixedHeight(20)
        self.lbl_hover.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.addWidget(self.lbl_hover)

        layout.addSpacing(5); line_flag = QFrame(); line_flag.setFrameShape(QFrame.HLine); layout.addWidget(line_flag); layout.addSpacing(5)

        layout.addWidget(QLabel("<b>🚩 Flag Suspicious Cell:</b> click a cell point, then Pin"))
        self.lbl_last_click = QLabel("尚未点击任何细胞")
        self.lbl_last_click.setStyleSheet("color: #888; font-size: 11px;")
        self.lbl_last_click.setWordWrap(True)
        layout.addWidget(self.lbl_last_click)

        h_flag_btns = QHBoxLayout()
        btn_pin = QPushButton("📌 Pin"); btn_export_flags = QPushButton("💾 Export"); btn_clear_flags = QPushButton("🗑 Clear")
        btn_pin.clicked.connect(self.pin_last_clicked_cell)
        btn_export_flags.clicked.connect(self.export_flagged_cells)
        btn_clear_flags.clicked.connect(self.clear_flagged_cells)
        h_flag_btns.addWidget(btn_pin); h_flag_btns.addWidget(btn_export_flags); h_flag_btns.addWidget(btn_clear_flags)
        layout.addLayout(h_flag_btns)

        self.lbl_flag_count = QLabel("🚩 已标记 0 个细胞")
        self.lbl_flag_count.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self.lbl_flag_count)

        layout.addSpacing(5); line2 = QFrame(); line2.setFrameShape(QFrame.HLine); layout.addWidget(line2); layout.addSpacing(5)

        layout.addWidget(QLabel("<b>2. Cell Class:</b>"))
        self.class_stack = QStackedWidget()

        # Stats mode: single class selector, populated dynamically once the
        # cell classes (marker combinations) are discovered from the data.
        # A checkable list (instead of a dropdown) keeps ~5 classes visible
        # at once with no extra click to open a menu; checking an entry
        # unchecks the rest so selection stays single, like the old combo.
        page_stats = QWidget(); layout_stats = QVBoxLayout(page_stats); layout_stats.setContentsMargins(0,0,0,0)
        self.list_class_single = QListWidget()
        self.list_class_single.setSelectionMode(QAbstractItemView.NoSelection)
        row_height = self.list_class_single.fontMetrics().height() + 10
        self.list_class_single.setFixedHeight(row_height * 5 + 2 * self.list_class_single.frameWidth())
        self.list_class_single.itemChanged.connect(self.on_class_single_item_changed)
        layout_stats.addWidget(self.list_class_single)
        self.class_stack.addWidget(page_stats)

        # Sample view mode: one checkbox per discovered cell class (no
        # symbol/shape mapping -- each class is just its own napari layer).
        # Populated dynamically in process_loaded_data(); container kept so it
        # can be cleared and rebuilt when data is (re)loaded.
        page_sample = QWidget()
        self.layout_sample_checkboxes = QVBoxLayout(page_sample)
        self.layout_sample_checkboxes.setContentsMargins(0,0,0,0)
        self.class_stack.addWidget(page_sample)
        layout.addWidget(self.class_stack); layout.addSpacing(10)

        layout.addWidget(QLabel("<b>3. Metric Type (Stats Only):</b>"))
        self.combo_metric = QComboBox()
        self.combo_metric.addItems(["Count", "Percentage", "Density", "Volume"])
        self.combo_metric.currentTextChanged.connect(self.on_metric_change)
        layout.addWidget(self.combo_metric)

        layout.addWidget(QLabel("<b>4. Ontology Level (Stats Only):</b>"))
        self.combo_level = QComboBox()
        self.combo_level.currentTextChanged.connect(self.on_level_change)
        layout.addWidget(self.combo_level)

        layout.addSpacing(6); layout.addWidget(QLabel("<b>🧮 Log2 FoldChange:</b>"))

        # Horizontal, compact colorbar -- a tall vertical bar here used to eat
        # a large chunk of the (width-capped) side panel's vertical space, at
        # the expense of the Search Regions section above it.
        fig = Figure(figsize=(3.0, 0.5), facecolor='#262930')
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.45, top=0.95)
        ax = fig.add_subplot(111)
        grad = np.linspace(-2, 2, 256).reshape(1, -1)
        cmap_mpl = LinearSegmentedColormap.from_list("blue_black_red", ["blue", "black", "red"])
        ax.imshow(grad, aspect='auto', cmap=cmap_mpl, extent=[-2, 2, 0, 1])
        ax.set_yticks([]); ax.xaxis.tick_bottom(); ax.tick_params(colors='white', labelsize=8)
        for spine in ax.spines.values(): spine.set_color('white')
        canvas = FigureCanvasQTAgg(fig)
        canvas.setFixedHeight(48)

        layout.addWidget(canvas)
        layout.addStretch()
        self.viewer.window.add_dock_widget(dock, area='right', name="Control Panel")

    # --- 新增：处理用户点击“加载数据”按钮的逻辑 ---
    def process_loaded_data(self):
        CONFIG['parent_data_dir'] = self.line_dir.text().strip()
        CONFIG['stats_long_table'] = self.line_stats.text().strip()
        CONFIG['std_atlas_path'] = self.line_atlas.text().strip()
        CONFIG['ontology_json_path'] = self.line_json.text().strip()

        if not CONFIG['std_atlas_path'] or not CONFIG['ontology_json_path']:
            QMessageBox.warning(None, "Missing Files", "Atlas (.tif) 和 Ontology JSON 是必填项！\n请先选择这两个基础文件。")
            return

        self.viewer.status = "⏳ Loading Data... Please wait."

        # 重新加载数据
        self.ontology = OntologyTree(CONFIG['ontology_json_path'])
        self.samples = DataLoader.scan_samples(CONFIG['parent_data_dir'])
        self.classes = DataLoader.discover_classes(self.samples)
        self.all_stats_df = DataLoader.load_stats(CONFIG['stats_long_table'])

        self._rebuild_class_checkboxes()
        self._rebuild_class_combo()
        self._rebuild_level_combo()

        # 更新下拉菜单
        self.combo_sample.blockSignals(True) # 暂时屏蔽信号防止触发错误渲染
        self.combo_sample.clear()

        if not self.all_stats_df.empty:
            self.combo_sample.addItem("📊 Statistical Analysis")

        for name in self.samples:
            self.combo_sample.addItem(f"🐭 [Native] {name}")
            self.combo_sample.addItem(f"📍 [Atlas ] {name}")

        if self.combo_sample.count() == 0:
            self.combo_sample.addItem("未找到有效数据")
            QMessageBox.information(None, "Info", "未在指定文件夹中扫描到有效的样本数据。")
        else:
            self.combo_sample.setCurrentIndex(0)
            self.on_mode_change(self.combo_sample.currentText()) # 手动触发第一次渲染

        self.combo_sample.blockSignals(False)
        self.viewer.status = "✅ Data loaded successfully."

    def _rebuild_class_checkboxes(self):
        """(Re)build one checkbox per discovered cell class in Sample-view mode."""
        while self.layout_sample_checkboxes.count():
            item = self.layout_sample_checkboxes.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        self.cell_checkboxes = {}
        for name in self.classes:
            cb = QCheckBox(name)
            cb.setChecked(True)
            cb.stateChanged.connect(lambda state, n=name: self.on_cell_check_toggle(n, state))
            self.layout_sample_checkboxes.addWidget(cb)
            self.cell_checkboxes[name] = cb

    def _rebuild_class_combo(self):
        """Stats-mode class selector. Union of folder-discovered classes and
        class_name values found in the loaded stats table, since merged
        pseudo-classes (e.g. 'glia', 'gfp' from stats_group_compare.py's
        merged_classes config) are summed at analysis time and have no
        matching cell_registration/ folder of their own."""
        self.list_class_single.blockSignals(True)
        self.list_class_single.clear()
        stats_classes = set(self.all_stats_df['class_name'].unique()) if not self.all_stats_df.empty else set()
        combined = sorted(set(self.classes) | stats_classes)
        for i, name in enumerate(combined):
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if i == 0 else Qt.Unchecked)
            self.list_class_single.addItem(item)
        self.list_class_single.blockSignals(False)
        self.current_class = combined[0] if combined else None

    def _rebuild_level_combo(self):
        self.combo_level.blockSignals(True)
        self.combo_level.clear()
        if not self.all_stats_df.empty:
            levels = sorted(int(l) for l in self.all_stats_df['level'].unique())
        elif self.ontology is not None:
            levels = list(range(self.ontology.max_level + 1))
        else:
            levels = []
        for lvl in levels:
            self.combo_level.addItem(str(lvl))
        self.combo_level.blockSignals(False)
        self.current_level = levels[0] if levels else 0

    # 下方保留原有的事件处理函数
    def on_point_size_change(self, val):
        for layer in self.viewer.layers:
            if layer.name.startswith("Cell:") or layer.name == ">> Highlight Cells <<":
                layer.size = val
            elif layer.name == "🚩 Flagged Cells":
                layer.size = val * 2.2

    def on_mode_change(self, text):
        if not self.ontology or "等待" in text or "未找" in text: return
        
        if "Statistical" in text or "Stats" in text:
            self.mode = "Stats"
            self.class_stack.setCurrentIndex(0)
            self.combo_metric.setEnabled(True)
            self.load_standard_view()
        else:
            self.class_stack.setCurrentIndex(1)
            self.combo_metric.setEnabled(False)
            sample_name = text.split('] ', 1)[1]
            if "[Native]" in text:
                self.mode = "Native"
                self.load_sample_native_view(sample_name)
            elif "[Atlas ]" in text:
                self.mode = "Atlas_Sample"
                self.load_sample_atlas_view(sample_name)

    def on_class_single_item_changed(self, item):
        """Keep the stats-mode class list single-select: checking an entry
        unchecks all others, mirroring the old combo box's behavior."""
        self.list_class_single.blockSignals(True)
        if item.checkState() == Qt.Checked:
            for i in range(self.list_class_single.count()):
                other = self.list_class_single.item(i)
                if other is not item:
                    other.setCheckState(Qt.Unchecked)
            self.current_class = item.text()
        else:
            # Prevent unchecking the only selected item -- there must
            # always be exactly one class selected in stats mode.
            item.setCheckState(Qt.Checked)
        self.list_class_single.blockSignals(False)
        self.refresh_heatmaps()

    def on_level_change(self, text):
        if not text: return
        self.current_level = int(text)
        self.refresh_heatmaps()

    def on_cell_check_toggle(self, name, state):
        if self.mode == "Stats": return
        layer_name = f"Cell: {name}"
        for layer in self.viewer.layers:
            if layer.name == layer_name:
                layer.visible = (state == Qt.Checked)
                break
        self.perform_search() 

    def on_metric_change(self, text):
        self.current_metric = text
        self.refresh_heatmaps()

if __name__ == "__main__":
    viewer = napari.Viewer(title="Spatial Explorer")
    controller = MainController(viewer)
    napari.run()