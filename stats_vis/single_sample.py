import napari
import pandas as pd
import numpy as np
import tifffile
import SimpleITK as sitk
import os
import glob
import json
import re


from PyQt5.QtWidgets import (QComboBox, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QFrame,
                             QCheckBox, QLineEdit, QPushButton, QDoubleSpinBox, QScrollArea,
                             QFileDialog, QMessageBox)
from PyQt5.QtCore import Qt

# ================= ⚙️ 用户配置区域 =================

CONFIG = {
    # 1. 单个样本的工作目录 (即你提到的 cell_registration 文件夹)
    "sample_dir": "/Users/fengyiyu/Downloads/Glab/clearmap/TSC_brains/s8", 
    
    # 2. 标准 Atlas 文件 (annotation_25.nrrd 或 tif)
    "std_atlas_path": "/Users/fengyiyu/Downloads/Glab/clearmap/DeMBA_p5/DeMBA_P5_annotation_trimmed_1_3_2__285-510_full_full.tif",

    # 3. JSON 格式脑区字典 (包含 graph_order 和 id)
    "ontology_json_path": "/Users/fengyiyu/Downloads/Glab/clearmap/DeMBA_p5/CCF_v3_ontology.json", 

    # 4. 视图旋转 (0=不旋转, 1=逆时针90°, 2=180°, 3=顺时针90°)
    "view_rotate_k": 0,
}

# ================= 🧠 1. JSON 脑区层级管理器 =================

class OntologyManager:
    def __init__(self, json_path):
        self.id_to_name = {}
        self.name_to_id = {}
        self.parse_ontology(json_path)

    def parse_ontology(self, json_path):
        if not os.path.exists(json_path): 
            print(f"❌ JSON not found: {json_path}")
            return
        
        print(f"📖 Parsing JSON Ontology...")
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        def extract_node(node):
            if isinstance(node, list):
                for item in node: extract_node(item)
                return
            if isinstance(node, dict):
                node_id = node.get('id') or node.get('structure_id')
                graph_order = node.get('graph_order') 
                node_name = node.get('name') or node.get('safe_name') or node.get('acronym')
                
                if node_name is not None:
                    s_name = DataLoader.clean_part(node_name)
                    if graph_order is not None:
                        self.id_to_name[int(graph_order)] = s_name
                        self.name_to_id[s_name] = int(graph_order)
                    if node_id is not None:
                        self.id_to_name[int(node_id)] = s_name
                        if s_name not in self.name_to_id:
                            self.name_to_id[s_name] = int(node_id)
                
                children = node.get('children') or node.get('msg')
                if children: extract_node(children)
        
        if isinstance(data, dict):
            if 'msg' in data: extract_node(data['msg'])
            elif 'children' in data: extract_node(data['children'])
            else: extract_node(data)
        elif isinstance(data, list):
            extract_node(data)

    def get_name(self, region_id):
        return self.id_to_name.get(region_id, f"Region {region_id}")

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
    def rotate_vol(arr, k):
        return np.rot90(arr, k=k, axes=(1, 2))

    @staticmethod
    def rotate_pts(pts, k, orig_shape):
        """将 (z,y,x) 坐标变换到旋转后的体积空间"""
        _, H, W = orig_shape
        pts = pts.copy()
        y, x = pts[:, 1].copy(), pts[:, 2].copy()
        if k == 1:   # 逆时针 90°: (y,x) → (W-1-x, y)
            pts[:, 1] = W - 1 - x
            pts[:, 2] = y
        elif k == 2: # 180°
            pts[:, 1] = H - 1 - y
            pts[:, 2] = W - 1 - x
        elif k == 3: # 顺时针 90°
            pts[:, 1] = x
            pts[:, 2] = H - 1 - y
        return pts

    @staticmethod
    def load_volume(path, dtype=None):
        """Read a 3D volume regardless of format: .tif/.tiff via tifffile,
        everything else (.nii/.nii.gz/.mhd/.mha/.nrrd) via SimpleITK -- ANTs
        writes standard NIfTI, and sitk reads it through the same ITK IO
        layer it already uses for ClearMap's .mhd files, so no special
        handling is needed per format beyond picking the reader."""
        if not path or not os.path.exists(path): return None
        if path.lower().endswith(('.tif', '.tiff')):
            arr = tifffile.imread(path)
        else:
            arr = sitk.GetArrayFromImage(sitk.ReadImage(path))
        return arr.astype(dtype) if dtype is not None else arr

    @staticmethod
    def resolve_native_paths(target_dir):
        """Find the raw sample image + atlas-labels-warped-to-sample-space
        file, whichever pipeline produced them:
          ClearMap: resampled.tif + volume/result.mhd
          ANTs:     <name>_fine_*um.nii.gz + <name>_labels_in_sample.nii.gz
        Returns (img_path, labels_path), either may be None if not found.
        """
        cm_img = os.path.join(target_dir, 'resampled.tif')
        cm_labels = os.path.join(target_dir, 'volume', 'result.mhd')
        if os.path.exists(cm_img) or os.path.exists(cm_labels):
            return (cm_img if os.path.exists(cm_img) else None,
                    cm_labels if os.path.exists(cm_labels) else None)

        fine_matches = glob.glob(os.path.join(target_dir, '*_fine_*um.nii.gz'))
        labels_matches = glob.glob(os.path.join(target_dir, '*_labels_in_sample.nii.gz'))
        return (fine_matches[0] if fine_matches else None,
                labels_matches[0] if labels_matches else None)

    @staticmethod
    def normalize_image_8bit(img_path):
        img = DataLoader.load_volume(img_path)
        if img is None: return None, None
        low, high = np.percentile(img, [0.5, 95.5])
        img_clipped = np.clip(img, low, high)
        return ((img_clipped - low) / (high - low) * 255).astype(np.uint8), img.shape

    @staticmethod
    def load_cells_from_registration(folder_path, ontology, coord_start):
        """动态扫描 cell_registration 下的子文件夹 (class_name)，读取其中的 cell_registration.csv

        列布局 (0-indexed): 0-2 原始像素坐标, 3-5 resample 空间坐标, 6-8 atlas 空间坐标,
        9 mapped_id, 10 region name, 11 slice_name, 12 tile_name, 13 score
        (后三者仅在 cellMap.py 较新版本生成的文件中存在)。coord_start 指定坐标起始列
        (3=resample空间, 6=atlas空间)。
        """
        all_dfs = []
        if not os.path.exists(folder_path): return pd.DataFrame()

        for entry in os.scandir(folder_path):
            if not entry.is_dir(): continue
            class_name = entry.name # 文件夹名即细胞分类名
            csv_path = os.path.join(entry.path, 'cell_registration.csv')

            if os.path.exists(csv_path):
                try:
                    df_raw = pd.read_csv(csv_path, header=None, names=range(20), engine='python')
                    if len(df_raw) > 0:
                        coords = df_raw.iloc[:, coord_start:coord_start + 3].values.astype(float)
                        valid_mask = ~np.isnan(coords).any(axis=1)
                        coords = coords[valid_mask]

                        ids = df_raw.iloc[:, 9].values[valid_mask]
                        ids = pd.to_numeric(pd.Series(ids), errors='coerce').fillna(0).astype(int).values

                        napari_pts = coords[:, [2, 1, 0]]

                        df_clean = pd.DataFrame(napari_pts, columns=['z', 'y', 'x'])
                        df_clean['class_name'] = class_name
                        df_clean['mapped_id'] = ids
                        df_clean['region'] = [ontology.get_name(uid) for uid in ids]

                        # 原始像素坐标 (0-2列) 始终存在，用于标记细胞后回溯原始 tile。
                        raw_xyz = df_raw.iloc[:, 0:3].values.astype(float)[valid_mask]
                        df_clean['raw_x'] = raw_xyz[:, 0]
                        df_clean['raw_y'] = raw_xyz[:, 1]
                        df_clean['raw_z'] = raw_xyz[:, 2]

                        # slice_name/tile_name/score 只在新版 cellMap.py 输出的
                        # cell_registration.csv 中存在 (11/12/13列)；旧文件留空。
                        if df_raw.shape[1] > 12 and df_raw.iloc[:, 12].notna().any():
                            df_clean['slice_name'] = df_raw.iloc[:, 11].values[valid_mask]
                            df_clean['tile_name'] = df_raw.iloc[:, 12].values[valid_mask]
                            df_clean['score'] = (pd.to_numeric(df_raw.iloc[:, 13], errors='coerce').values[valid_mask]
                                                  if df_raw.shape[1] > 13 else np.nan)
                        else:
                            df_clean['slice_name'] = np.nan
                            df_clean['tile_name'] = np.nan
                            df_clean['score'] = np.nan

                        all_dfs.append(df_clean)
                        print(f"✅ 成功加载 {len(df_clean)} 个 '{class_name}' 细胞。")
                except Exception as e:
                    print(f"❌ 解析 '{class_name}' 坐标出错: {e}")

        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


# ================= 🎮 3. 主控制器 =================

class MainController:
    def __init__(self, viewer):
        self.viewer = viewer

        self.viewer.axes.visible = False

        self.ontology = OntologyManager(CONFIG['ontology_json_path'])
        
        # 获取基础工作目录信息
        self.target_dir = CONFIG['sample_dir']
        self.sample_name = os.path.basename(os.path.normpath(self.target_dir))
        
        self.current_atlas_labels = None
        self.current_cells_df = pd.DataFrame() 
        self.highlight_atlas = None
        self.highlight_cells = None
        self.last_hover_val = -1
        
        self.mode = None
        self.cell_checkboxes = {}
        self.last_search_mode = "Exact"

        # 🚩 Flagging: click a cell -> pin it -> export tile/slice provenance
        # for suspicious cells so the user can pull up the raw TB-scale tile.
        self.flagged_cells = []       # list[dict], persists across Native/Atlas views
        self.last_clicked_cell = None # dict, most recent cell click (candidate to pin)
        self.flag_layer = None        # current napari points layer showing pins for this view
        self.last_highlighted_df = pd.DataFrame()  # last Search Regions result, for bulk export

        self.setup_ui()
        self.setup_callbacks()
        
        # 默认自动加载 Native 视图
        self.on_mode_change(self.combo_sample.currentText())

    def setup_highlight_layers(self, shape):
        for name in [">> Highlight Atlas <<", ">> Highlight Cells <<", "✨ Selection"]:
            if name in self.viewer.layers: self.viewer.layers.remove(name)
                
        self.highlight_atlas = self.viewer.add_labels(np.zeros(shape, dtype=np.uint32), name=">> Highlight Atlas <<", opacity=0.8)
        self.highlight_cells = self.viewer.add_points(np.empty((0, 3)), ndim=3, name=">> Highlight Cells <<",
                                                      face_color='white', border_color='yellow', size=self.spin_point_size.value(), opacity=1.0)

    def render_cells_from_df(self, df_cells, labels_layer):
        if df_cells.empty or labels_layer is None: return

        # --- 新增：如果未勾选，则过滤掉 mapped_id 为 0 的背景细胞 ---
        if hasattr(self, 'cb_show_bg') and not self.cb_show_bg.isChecked():
            df_cells = df_cells[df_cells['mapped_id'] != 0]
        # -------------------------------------------------------------

        id_color_map = {0: np.array([0.5, 0.5, 0.5, 1.0])}
        for uid in df_cells['mapped_id'].unique():
            if uid != 0: id_color_map[uid] = labels_layer.get_color(uid)

        # 动态读取并渲染独特的 class_name
        unique_classes = sorted(df_cells['class_name'].unique())
        
        for cls_name in unique_classes:
            sub_df = df_cells[df_cells['class_name'] == cls_name]
            if len(sub_df) > 0:
                coords = sub_df[['z', 'y', 'x']].values
                colors = np.array([id_color_map[uid] for uid in sub_df['mapped_id']])
                
                is_vis = self.cell_checkboxes[cls_name].isChecked() if cls_name in self.cell_checkboxes else True
                
                layer = self.viewer.add_points(
                    coords, name=f"Cell: {cls_name}", face_color=colors,
                    symbol='disc', size=self.spin_point_size.value(), border_width=0, blending='translucent', visible=is_vis
                )
                # Carried through for click-to-pin export (tile/slice
                # provenance -- see DataLoader.load_cells_from_registration).
                layer.features = pd.DataFrame({
                    'Region': sub_df['region'].values,
                    'Class': sub_df['class_name'].values,
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
        
        # 1. 过滤 Atlas
        matched_ids = []
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
            
        # 2. 过滤 Cells
        if not self.current_cells_df.empty:
            active_classes = [name for name, cb in self.cell_checkboxes.items() if cb.isChecked()]
            if search_mode == 'Exact':
                region_mask = self.current_cells_df['region'].str.lower() == keyword.lower()
            else:
                region_mask = self.current_cells_df['region'].str.contains(keyword, case=False, regex=False)
                
            class_mask = self.current_cells_df['class_name'].isin(active_classes)
            subset_df = self.current_cells_df[region_mask & class_mask]
            
            # --- 新增：搜索高亮时同步排除背景细胞 ---
            if hasattr(self, 'cb_show_bg') and not self.cb_show_bg.isChecked():
                subset_df = subset_df[subset_df['mapped_id'] != 0]
            # ----------------------------------------

            subset_points = subset_df[['z', 'y', 'x']].values

            self.highlight_cells.data = subset_points if len(subset_points) > 0 else np.empty((0, 3))
            self.last_highlighted_df = subset_df.copy()
            self.viewer.status = f"✅ [{search_mode}] Found {len(matched_ids)} regions | Cells: {len(subset_points)}"

    def update_class_filter_ui(self, df_cells):
        """核心：根据读取到的 df 动态生成 CheckBox"""
        for i in reversed(range(self.layout_classes.count())): 
            widget = self.layout_classes.itemAt(i).widget()
            if widget: 
                widget.setParent(None)
                widget.deleteLater()
        self.cell_checkboxes.clear()

        if df_cells.empty: return

        unique_classes = sorted(df_cells['class_name'].unique())

        for cls_name in unique_classes:
            cb = QCheckBox(cls_name)
            cb.setChecked(True)
            cb.stateChanged.connect(lambda state, n=cls_name: self.on_cell_check_toggle(n, state))
            self.layout_classes.addWidget(cb)
            self.cell_checkboxes[cls_name] = cb

    # --- 视图加载 ---

    def load_sample_native_view(self):
        self.viewer.layers.clear()
        self.current_cells_df = pd.DataFrame()
        print(f"\n🚀 Loading Native View from: {self.target_dir}")
        
        resampled_path, mhd_path = DataLoader.resolve_native_paths(self.target_dir)
        cell_reg_dir = os.path.join(self.target_dir, 'cell_registration')

        img_norm, shape = DataLoader.normalize_image_8bit(resampled_path)
        mhd = DataLoader.load_volume(mhd_path, dtype=np.uint32)
        # 使用 cell_registration.csv 中已经算好的 resample 空间坐标 (第4-6列, 0-indexed 3:6)
        df_cells = DataLoader.load_cells_from_registration(cell_reg_dir, self.ontology, coord_start=3)

        k = CONFIG.get('view_rotate_k', 0)
        if k:
            if img_norm is not None:
                img_norm = DataLoader.rotate_vol(img_norm, k)
            if mhd is not None:
                orig_shape = mhd.shape
                mhd = DataLoader.rotate_vol(mhd, k)
                if not df_cells.empty:
                    df_cells[['z', 'y', 'x']] = DataLoader.rotate_pts(df_cells[['z', 'y', 'x']].values, k, orig_shape)

        if img_norm is not None:
            self.viewer.add_image(img_norm, name="Raw Image", colormap="gray", blending='additive')
        else:
            print(f"⚠️ 找不到样本图像 (resampled.tif 或 *_fine_*um.nii.gz)，目录: {self.target_dir}")

        labels_layer = None
        if mhd is not None:
            self.current_atlas_labels = mhd
            labels_layer = self.viewer.add_labels(mhd, name="Atlas Regions", opacity=0.05, visible=False)
            self.setup_highlight_layers(mhd.shape)
        else:
            print(f"⚠️ 找不到样本空间标签图 (volume/result.mhd 或 *_labels_in_sample.nii.gz)，无法为细胞赋予准确的脑区颜色和筛选。")

        self.current_cells_df = df_cells
        self.update_class_filter_ui(df_cells)
        self.render_cells_from_df(df_cells, labels_layer)
        self._apply_grid_layout("Atlas Regions")
        self.setup_flag_layer()

    def load_sample_atlas_view(self):
        self.viewer.layers.clear()
        self.current_cells_df = pd.DataFrame()
        print(f"\n🚀 Loading Atlas Space View from: {self.target_dir}")

        atlas_layer = None
        atlas_path = CONFIG['std_atlas_path']
        cell_reg_dir = os.path.join(self.target_dir, 'cell_registration')
        # 使用 cell_registration.csv 中的 atlas 空间坐标 (第7-9列, 0-indexed 6:9)
        df_cells = DataLoader.load_cells_from_registration(cell_reg_dir, self.ontology, coord_start=6)

        k = CONFIG.get('view_rotate_k', 0)

        # ANTs pipeline 额外产出了"样本变形到图谱空间"的灰度图 (<name>_in_atlas.nii.gz)，
        # 跟 Atlas Anatomy 标签图是同一个网格，直接叠加显示，方便肉眼核对配准准不准。
        # ClearMap 原生流程没有这个文件，找不到就跳过，不影响原有功能。
        in_atlas_matches = glob.glob(os.path.join(self.target_dir, '*_in_atlas.nii.gz'))
        if in_atlas_matches:
            img_norm, _ = DataLoader.normalize_image_8bit(in_atlas_matches[0])
            if img_norm is not None:
                if k: img_norm = DataLoader.rotate_vol(img_norm, k)
                self.viewer.add_image(img_norm, name="Sample in Atlas", colormap="gray", blending='additive')

        if os.path.exists(atlas_path):
            data = DataLoader.load_volume(atlas_path, dtype=np.uint32)
            if k:
                orig_shape = data.shape
                data = DataLoader.rotate_vol(data, k)
                if not df_cells.empty:
                    df_cells[['z', 'y', 'x']] = DataLoader.rotate_pts(df_cells[['z', 'y', 'x']].values, k, orig_shape)
            self.current_atlas_labels = data
            atlas_layer = self.viewer.add_labels(self.current_atlas_labels, name="Atlas Anatomy", opacity=0.3)
            self.setup_highlight_layers(self.current_atlas_labels.shape)

        self.current_cells_df = df_cells
        self.update_class_filter_ui(df_cells)
        self.render_cells_from_df(df_cells, atlas_layer)
        self._apply_grid_layout("Atlas Anatomy")
        self.setup_flag_layer()

    def _apply_grid_layout(self, atlas_layer_name):
        if atlas_layer_name in self.viewer.layers:
            idx = self.viewer.layers.index(atlas_layer_name)
            self.viewer.layers.move(idx, -1)
        self.viewer.grid.enabled = True
        self.viewer.grid.shape = (-1, 3)

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
                'sample': self.sample_name,
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

    def setup_flag_layer(self):
        """(Re)create the '🚩 Flagged Cells' points layer for the current
        view mode, restoring any previously-pinned points so flags survive
        switching between Native/Atlas (self.flagged_cells is the durable
        store; this layer is just its visual projection onto the current
        view's coordinate space)."""
        if "🚩 Flagged Cells" in self.viewer.layers:
            self.viewer.layers.remove("🚩 Flagged Cells")
        matches = [f for f in self.flagged_cells if f['mode'] == self.mode]
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
        export_df = df.copy()
        export_df['sample'] = self.sample_name
        cols = ['sample', 'class_name', 'region', 'mapped_id', 'tile_name', 'slice_name',
                'raw_x', 'raw_y', 'raw_z', 'z', 'y', 'x', 'score']
        cols = [c for c in cols if c in export_df.columns]
        export_df[cols].to_csv(path, index=False)
        QMessageBox.information(None, "Exported", f"已导出 {len(export_df)} 个细胞 → {path}")

    def setup_callbacks(self):
        @self.viewer.mouse_move_callbacks.append
        def on_mouse_move(viewer, event):
            if self.current_atlas_labels is None: return
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
                            self.lbl_hover.setWordWrap(True)  
                            self.lbl_hover.setFixedHeight(40) 
                            self.lbl_hover.setAlignment(Qt.AlignTop | Qt.AlignLeft) 
                            viewer.status = f"🧠 {region_name} (ID: {val})"
                        else:
                            self.lbl_hover.setText("📍 Hover: Background")
                            viewer.status = ""

        @self.viewer.mouse_drag_callbacks.append
        def on_click(viewer, event):
            active_layer = viewer.layers.selection.active
            if event.type != 'mouse_press' or self.current_atlas_labels is None: return
            if active_layer is not None and active_layer.mode == 'pan_zoom': return
            
            c = np.round(viewer.cursor.position).astype(int)
            shape = self.current_atlas_labels.shape
            if not all(0 <= c[i] < shape[i] for i in range(3)): return
            
            rid = self.current_atlas_labels[c[0], c[1], c[2]]
            if rid > 0: 
                name = self.ontology.get_name(rid)
                print(f"\n📍 Clicked Region: {name} (ID: {rid})")
                
                if name and not name.startswith("Region"):
                    self.input_search.setText(name)
                    self.perform_search("Exact")

    def setup_ui(self):
        dock = QWidget()
        dock.setMaximumWidth(320)
        layout = QVBoxLayout(dock)
        
        layout.addWidget(QLabel("<b>1. Workspace Mode:</b>"))
        self.combo_sample = QComboBox()
        # 直接使用解析好的当前文件夹名称进行展示
        self.combo_sample.addItem(f"🐭 [Native] {self.sample_name}")
        self.combo_sample.addItem(f"📍 [Atlas ] {self.sample_name}")
        self.combo_sample.currentTextChanged.connect(self.on_mode_change)
        layout.addWidget(self.combo_sample)

        h_size = QHBoxLayout()
        h_size.addWidget(QLabel("<b>Cell Size:</b>"))
        self.spin_point_size = QDoubleSpinBox()
        self.spin_point_size.setRange(0.1, 50.0)
        self.spin_point_size.setValue(5.0)
        self.spin_point_size.setSingleStep(1.0)
        self.spin_point_size.valueChanged.connect(self.on_point_size_change)
        h_size.addWidget(self.spin_point_size)
        layout.addLayout(h_size)

        layout.addSpacing(5); line1 = QFrame(); line1.setFrameShape(QFrame.HLine); layout.addWidget(line1); layout.addSpacing(5)

        layout.addWidget(QLabel("<b>🔍 Search Regions:</b>"))
        self.input_search = QLineEdit(); self.input_search.setPlaceholderText("Region name...")
        self.input_search.returnPressed.connect(lambda: self.perform_search())
        layout.addWidget(self.input_search)
        
        h_search_btns = QHBoxLayout()
        btn_fuzzy = QPushButton("Fuzzy Search")
        btn_exact = QPushButton("Exact Search")
        btn_fuzzy.clicked.connect(lambda: self.perform_search("Fuzzy"))
        btn_exact.clicked.connect(lambda: self.perform_search("Exact"))
        h_search_btns.addWidget(btn_fuzzy)
        h_search_btns.addWidget(btn_exact)
        layout.addLayout(h_search_btns)

        btn_export_highlight = QPushButton("📤 Export Highlighted Cells (tile/slice)")
        btn_export_highlight.clicked.connect(self.export_highlighted_cells)
        layout.addWidget(btn_export_highlight)

        self.lbl_hover = QLabel("📍 Hover: None")
        self.lbl_hover.setStyleSheet("color: #888; font-size: 11px;")
        self.lbl_hover.setWordWrap(True)
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

        layout.addWidget(QLabel("<b>2. Cell Class Filter:</b>"))

        # --- 新增：控制是否显示背景细胞的 CheckBox ---
        self.cb_show_bg = QCheckBox("Show Background Cells (ID=0)")
        self.cb_show_bg.setChecked(False) # 默认不显示背景细胞
        self.cb_show_bg.stateChanged.connect(self.on_bg_toggle)
        layout.addWidget(self.cb_show_bg)
        # ----------------------------------------
        
        # 动态 CheckBox 的容器
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        self.class_checkbox_container = QWidget()
        self.layout_classes = QVBoxLayout(self.class_checkbox_container)
        self.layout_classes.setContentsMargins(0,0,0,0)
        scroll.setWidget(self.class_checkbox_container)
        
        layout.addWidget(scroll)
        self.viewer.window.add_dock_widget(dock, area='right', name="Control Panel")

    def on_point_size_change(self, val):
        for layer in self.viewer.layers:
            if layer.name.startswith("Cell:") or layer.name == ">> Highlight Cells <<":
                layer.size = val
            elif layer.name == "🚩 Flagged Cells":
                layer.size = val * 2.2

    def on_mode_change(self, text):
        if not text: return
        if "[Native]" in text:
            self.mode = "Native"
            self.load_sample_native_view()
        elif "[Atlas ]" in text:
            self.mode = "Atlas_Sample"
            self.load_sample_atlas_view()
    
    def on_cell_check_toggle(self, name, state):
        layer_name = f"Cell: {name}"
        for layer in self.viewer.layers:
            if layer.name == layer_name:
                layer.visible = (state == Qt.Checked)
                break
        self.perform_search() 

    def on_bg_toggle(self, state):
        """当勾选/取消背景细胞显示时触发，重新渲染细胞图层"""
        # 1. 移除旧的细胞图层
        layers_to_remove = [layer.name for layer in self.viewer.layers if layer.name.startswith("Cell:")]
        for name in layers_to_remove:
            self.viewer.layers.remove(name)
            
        # 2. 找到当前的 Atlas 标签图层（用于赋予对应的脑区颜色）
        labels_layer = None
        for layer in self.viewer.layers:
            if layer.name in ["Atlas Regions", "Atlas Anatomy"]:
                labels_layer = layer
                break
                
        # 3. 根据当前设置重新渲染并恢复搜索高亮
        self.render_cells_from_df(self.current_cells_df, labels_layer)
        self.perform_search()

if __name__ == "__main__":
    viewer = napari.Viewer(title="Spatial Explorer (Single Workspace)")
    controller = MainController(viewer)
    napari.run()