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
                             QFileDialog, QMessageBox, QGroupBox)
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
    "stats_excel": "",
    "std_atlas_path": "",
    "ontology_json_path": "", 

    # 5. 分辨率参数 [X, Y, Z] (单位: um)
    "res_raw":   np.array([0.65, 0.65, 20.0]), 
    "res_atlas": np.array([20.0, 20.0, 20.0]), # 20um for p5 atlas

    # 6. 类别名称映射
    "labels_to_names": {
        0: "red glia", 1: "green glia", 2: "yellow glia",
        3: "red neuron", 4: "green neuron", 5: "yellow neuron"
    },
    
    # 7. 细胞形状映射
    "class_symbols": {
        0: 'disc', 1: 'square', 2: 'triangle_up', 
        3: 'triangle_down', 4: 'star', 5: 'cross'
    }
}

# ================= 🧠 1. JSON 脑区层级管理器 =================
class OntologyManager:
    def __init__(self, json_path):
        self.id_to_name = {}
        self.name_to_id = {}
        if json_path and os.path.exists(json_path):
            self.parse_ontology(json_path)

    def parse_ontology(self, json_path):
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
    def scan_samples(parent_dir):
        samples = {}
        if not parent_dir or not os.path.exists(parent_dir): return samples
        for entry in os.scandir(parent_dir):
            if not entry.is_dir(): continue
            name = entry.name
            group = 'Experimental (FF)' if name.startswith('ff') else ('Control (FW)' if name.startswith('fw') else None)
            if not group: continue
            
            res_path = os.path.join(entry.path, 'resampled.tif')
            mhd_path = os.path.join(entry.path, 'volume', 'result.mhd')
            
            if os.path.exists(res_path):
                samples[name] = {
                    'path': entry.path, 'group': group,
                    'resampled': res_path, 'mhd': mhd_path,
                    'cell_dir_raw': os.path.join(entry.path, 'cell_centroids'),
                    'cell_dir_reg': os.path.join(entry.path, 'cell registration')
                }
        return samples

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
    def load_cells_native_df(folder_path, raw_res, target_res, mhd_data, ontology):
        all_dfs = []
        scale_factor = raw_res / target_res 
        mhd_shape = mhd_data.shape if mhd_data is not None else (0,0,0)

        for i, class_name in CONFIG['labels_to_names'].items():
            csv_path = os.path.join(folder_path, f'ob_{i}.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path, header=None)
                if len(df) == 0: continue
                napari_pts = (df.values * scale_factor)[:, [2, 1, 0]] 
                
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
                all_dfs.append(df_clean)
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    @staticmethod
    def load_cells_atlas_df(folder_path, ontology):
        all_dfs = []
        for i, class_name in CONFIG['labels_to_names'].items():
            csv_path = os.path.join(folder_path, str(i), 'cell_registration.csv')
            if os.path.exists(csv_path):
                try:
                    df_raw = pd.read_csv(csv_path, header=None, names=range(20), engine='python')
                    if len(df_raw) > 0:
                        coords = df_raw.iloc[:, 3:6].values.astype(float)
                        valid_mask = ~np.isnan(coords).any(axis=1)
                        coords = coords[valid_mask]

                        ids = df_raw.iloc[:, 6].values[valid_mask]
                        ids = pd.to_numeric(pd.Series(ids), errors='coerce').fillna(0).astype(int).values

                        napari_pts = coords[:, [2, 1, 0]]
                        
                        df_clean = pd.DataFrame(napari_pts, columns=['z', 'y', 'x'])
                        df_clean['class_name'] = class_name
                        df_clean['mapped_id'] = ids
                        df_clean['region'] = [ontology.get_name(uid) for uid in ids]
                        all_dfs.append(df_clean)
                except Exception as e:
                    print(f"❌ 解析 '{class_name}' 坐标出错: {e}")
        return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    @staticmethod
    def load_stats(excel_path):
        full_stats = {name: {} for name in CONFIG['labels_to_names'].values()}
        full_stats["Volume"] = {}
        if not excel_path or not os.path.exists(excel_path): return full_stats
        try:
            xls = pd.read_excel(excel_path, sheet_name=None, engine='openpyxl')
            for sheet_name, df in xls.items():
                clean_sheet = sheet_name.strip().lower().replace(" ", "_")
                metric_type = "Unknown"
                if "cnt" in clean_sheet or "count" in clean_sheet: metric_type = "Count"
                elif "den" in clean_sheet: metric_type = "Density"
                elif "pct" in clean_sheet or "percent" in clean_sheet: metric_type = "Percentage"
                elif "vol" in clean_sheet: metric_type = "Volume"
                
                target_class = "Volume" if metric_type == "Volume" else next(
                    (v for k, v in CONFIG['labels_to_names'].items() if v.strip().lower().replace(" ", "_") in clean_sheet), None)
                if not target_class: continue
                
                cols = {str(c).lower(): c for c in df.columns}
                c_id, c_fc, c_p_raw, c_p_fdr = [
                    next((cols[c] for c in cols if any(k in c for k in kw)), None) 
                    for kw in [['id', 'region_id'], ['fc'], ['p_val', 'pvalue'], ['fdr', 'q_val']]
                ]
                if not c_id: continue

                sheet_stats = {}
                for _, row in df.iterrows():
                    if pd.isna(row[c_id]): continue
                    try: 
                        rid = int(row[c_id])
                        sheet_stats[rid] = {
                            'FC': float(row[c_fc]) if c_fc and not pd.isna(row[c_fc]) else 0,
                            'P_Raw': -np.log10(float(row[c_p_raw]) + 1e-20) if c_p_raw and not pd.isna(row[c_p_raw]) else 0,
                            'P_FDR': -np.log10(float(row[c_p_fdr]) + 1e-20) if c_p_fdr and not pd.isna(row[c_p_fdr]) else 0
                        }
                    except ValueError: continue
                full_stats[target_class][metric_type] = sheet_stats
        except Exception as e: print(f"❌ Excel Error: {e}")
        return full_stats

# ================= 🎮 3. 主控制器 =================
class MainController:
    def __init__(self, viewer):
        self.viewer = viewer
        
        # 初始状态为空，等待用户点击加载
        self.ontology = None
        self.samples = {}
        self.all_stats = {}
        
        self.current_atlas_labels = None
        self.current_cells_df = pd.DataFrame() 
        self.highlight_atlas = None
        self.highlight_cells = None
        self.last_hover_val = -1
        
        self.mode = "Stats"
        self.current_class = CONFIG['labels_to_names'][0]
        self.current_metric = "Count"
        self.cell_checkboxes = {}
        self.last_search_mode = "Exact"

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

        for class_idx, cls_name in CONFIG['labels_to_names'].items():
            sub_df = df_cells[df_cells['class_name'] == cls_name]
            if len(sub_df) > 0:
                coords = sub_df[['z', 'y', 'x']].values
                colors = np.array([id_color_map[uid] for uid in sub_df['mapped_id']])
                is_vis = self.cell_checkboxes[cls_name].isChecked()
                
                symbol = CONFIG['class_symbols'].get(class_idx, 'disc')
                
                layer = self.viewer.add_points(
                    coords, name=f"Cell: {cls_name}", face_color=colors,
                    symbol=symbol, size=self.spin_point_size.value(), border_width=0, blending='translucent', visible=is_vis
                )
                layer.features = pd.DataFrame({'Region': sub_df['region'].values})
                layer.events.highlight.connect(self.on_cell_layer_click)

    def perform_search(self, search_mode=None):
        if search_mode is not None:
            self.last_search_mode = search_mode
        search_mode = self.last_search_mode

        keyword = self.input_search.text().strip()
        
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
            self.viewer.status = f"✅ [{search_mode}] Found {len(matched_ids)} regions | Cells: {len(subset_points)}"

    def load_standard_view(self):
        self.viewer.layers.clear()
        self.current_cells_df = pd.DataFrame()
        atlas_path = CONFIG['std_atlas_path']
        
        if atlas_path and os.path.exists(atlas_path):
            data = tifffile.imread(atlas_path) if atlas_path.lower().endswith(('.tif', '.tiff')) else __import__('nrrd').read(atlas_path)[0]
            self.current_atlas_labels = data.astype(np.uint32)
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
            labels_layer = self.viewer.add_labels(mhd, name="Atlas Regions", opacity=0.05, visible=False)
            self.setup_highlight_layers(mhd.shape)

        df_cells = DataLoader.load_cells_native_df(s['cell_dir_raw'], CONFIG['res_raw'], CONFIG['res_atlas'], mhd, self.ontology)
        self.current_cells_df = df_cells
        self.render_cells_from_df(df_cells, labels_layer)

    def load_sample_atlas_view(self, sample_key):
        self.viewer.layers.clear()
        self.current_cells_df = pd.DataFrame()
        s = self.samples[sample_key]

        atlas_layer = None
        atlas_path = CONFIG['std_atlas_path']
        
        if atlas_path and os.path.exists(atlas_path):
            data = tifffile.imread(atlas_path) if atlas_path.lower().endswith(('.tif', '.tiff')) else __import__('nrrd').read(atlas_path)[0]
            self.current_atlas_labels = data.astype(np.uint32)
            atlas_layer = self.viewer.add_labels(self.current_atlas_labels, name="Atlas Anatomy", opacity=0.3)
            self.setup_highlight_layers(self.current_atlas_labels.shape)
        
        df_cells = DataLoader.load_cells_atlas_df(s['cell_dir_reg'], self.ontology)
        self.current_cells_df = df_cells
        self.render_cells_from_df(df_cells, atlas_layer)

    def on_cell_layer_click(self, event):
        layer = event.source
        if self.viewer.layers.selection.active != layer: return
        if len(layer.selected_data) > 0:
            idx = list(layer.selected_data)[0]
            full_name = layer.features['Region'].iloc[idx]
            self.input_search.setText(full_name)
            self.perform_search("Exact")
            layer.selected_data = set() 

    def refresh_heatmaps(self):
        if self.mode != "Stats" or self.current_atlas_labels is None: return 
        for layer in list(self.viewer.layers):
            if "Stats:" in layer.name: self.viewer.layers.remove(layer)

        metric_stats = self.all_stats.get(self.current_class, {}).get(self.current_metric, {})
        if not metric_stats: return

        max_id = int(self.current_atlas_labels.max())
        lut_raw, lut_fdr = np.zeros(max_id + 1), np.zeros(max_id + 1)
        
        for rid, s in metric_stats.items():
            if rid > max_id: continue
            val = np.log2(s.get('FC', 0) + 1e-9) if s.get('FC', 0) > 0 else s.get('FC', 0)
            if s.get('P_Raw', 0) >= 1.3: lut_raw[rid] = val
            if s.get('P_FDR', 0) >= 1.3: lut_fdr[rid] = val

        dark_cmap = self.get_dark_colormap()
        self.viewer.add_image(lut_raw[self.current_atlas_labels], name=f"Stats: {self.current_metric} (Raw P)", colormap=dark_cmap, contrast_limits=[-2,2], blending='additive', visible=True)
        self.viewer.add_image(lut_fdr[self.current_atlas_labels], name=f"Stats: {self.current_metric} (FDR)", colormap=dark_cmap, contrast_limits=[-2,2], blending='additive', visible=False)

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
        def create_path_row(label, is_dir=False):
            h = QHBoxLayout()
            h.addWidget(QLabel(label))
            btn = QPushButton("Browse")
            line = QLineEdit()
            line.setReadOnly(True)
            h.addWidget(line)
            h.addWidget(btn)
            return h, line, btn

        r1, self.line_dir, btn_dir = create_path_row("Samples Dir:")
        r2, self.line_excel, btn_excel = create_path_row("Stats Excel:")
        r3, self.line_atlas, btn_atlas = create_path_row("Atlas (.tif):")
        r4, self.line_json, btn_json = create_path_row("Ontology JSON:")

        btn_dir.clicked.connect(lambda: self.line_dir.setText(QFileDialog.getExistingDirectory(dock, "Select Samples Directory")))
        btn_excel.clicked.connect(lambda: self.line_excel.setText(QFileDialog.getOpenFileName(dock, "Select Excel", "", "Excel Files (*.xlsx)")[0]))
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
        
        self.lbl_hover = QLabel("📍 Hover: None")
        self.lbl_hover.setStyleSheet("color: #888; font-size: 11px;")
        self.lbl_hover.setWordWrap(True); self.lbl_hover.setFixedHeight(20)
        self.lbl_hover.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        layout.addWidget(self.lbl_hover)

        layout.addSpacing(5); line2 = QFrame(); line2.setFrameShape(QFrame.HLine); layout.addWidget(line2); layout.addSpacing(5)

        layout.addWidget(QLabel("<b>2. Cell Class:</b>"))
        self.class_stack = QStackedWidget()
        
        page_stats = QWidget(); layout_stats = QVBoxLayout(page_stats); layout_stats.setContentsMargins(0,0,0,0)
        self.combo_class_single = QComboBox()
        for _, name in CONFIG['labels_to_names'].items(): self.combo_class_single.addItem(name)
        self.combo_class_single.addItem("Volume")
        self.combo_class_single.currentTextChanged.connect(self.on_class_single_change)
        layout_stats.addWidget(self.combo_class_single)
        self.class_stack.addWidget(page_stats)
        
        page_sample = QWidget(); layout_sample = QVBoxLayout(page_sample); layout_sample.setContentsMargins(0,0,0,0)
        shape_icon_map = {'disc': '●', 'square': '■', 'triangle_up': '▲', 'triangle_down': '▼', 'star': '★', 'cross': '✚'}
        for i, name in CONFIG['labels_to_names'].items():
            icon = shape_icon_map.get(CONFIG['class_symbols'].get(i, 'disc'), '●')
            cb = QCheckBox(f"{icon} {name}")
            cb.setChecked(True) 
            cb.stateChanged.connect(lambda state, n=name: self.on_cell_check_toggle(n, state))
            layout_sample.addWidget(cb)
            self.cell_checkboxes[name] = cb
            
        self.class_stack.addWidget(page_sample)
        layout.addWidget(self.class_stack); layout.addSpacing(10)
        
        layout.addWidget(QLabel("<b>3. Metric Type (Stats Only):</b>"))
        self.combo_metric = QComboBox()
        self.combo_metric.addItems(["Count", "Density", "Percentage", "Volume"])
        self.combo_metric.currentTextChanged.connect(self.on_metric_change)
        layout.addWidget(self.combo_metric)

        layout.addSpacing(10); layout.addWidget(QLabel("<b>🧮 Log2 FoldChange:</b>"))
        
        fig = Figure(figsize=(0.8, 3.0), facecolor='#262930')
        fig.subplots_adjust(left=0.1, right=0.4, bottom=0.05, top=0.95)
        ax = fig.add_subplot(111)
        grad = np.linspace(2, -2, 256).reshape(-1, 1) 
        cmap_mpl = LinearSegmentedColormap.from_list("blue_black_red", ["blue", "black", "red"])
        ax.imshow(grad, aspect='auto', cmap=cmap_mpl, extent=[0, 1, -2, 2])
        ax.set_xticks([]); ax.yaxis.tick_right(); ax.tick_params(colors='white', labelsize=8)
        canvas = FigureCanvasQTAgg(fig)
        
        h_leg = QHBoxLayout(); h_leg.addWidget(canvas, alignment=Qt.AlignLeft); h_leg.addStretch()
        layout.addLayout(h_leg)
        
        layout.addStretch()
        self.viewer.window.add_dock_widget(dock, area='right', name="Control Panel")

    # --- 新增：处理用户点击“加载数据”按钮的逻辑 ---
    def process_loaded_data(self):
        CONFIG['parent_data_dir'] = self.line_dir.text().strip()
        CONFIG['stats_excel'] = self.line_excel.text().strip()
        CONFIG['std_atlas_path'] = self.line_atlas.text().strip()
        CONFIG['ontology_json_path'] = self.line_json.text().strip()
        
        if not CONFIG['std_atlas_path'] or not CONFIG['ontology_json_path']:
            QMessageBox.warning(None, "Missing Files", "Atlas (.tif) 和 Ontology JSON 是必填项！\n请先选择这两个基础文件。")
            return
            
        self.viewer.status = "⏳ Loading Data... Please wait."
        
        # 重新加载数据
        self.ontology = OntologyManager(CONFIG['ontology_json_path'])
        self.samples = DataLoader.scan_samples(CONFIG['parent_data_dir'])
        self.all_stats = DataLoader.load_stats(CONFIG['stats_excel'])

        # 更新下拉菜单
        self.combo_sample.blockSignals(True) # 暂时屏蔽信号防止触发错误渲染
        self.combo_sample.clear()
        
        if self.all_stats and "Volume" in self.all_stats:
            self.combo_sample.addItem("📊 Statistical Analysis")
            
        for name, info in self.samples.items():
            self.combo_sample.addItem(f"🐭 [Native] {info['group']}: {name}")
            self.combo_sample.addItem(f"📍 [Atlas ] {info['group']}: {name}")
            
        if self.combo_sample.count() == 0:
            self.combo_sample.addItem("未找到有效数据")
            QMessageBox.information(None, "Info", "未在指定文件夹中扫描到有效的样本数据。")
        else:
            self.combo_sample.setCurrentIndex(0)
            self.on_mode_change(self.combo_sample.currentText()) # 手动触发第一次渲染
            
        self.combo_sample.blockSignals(False)
        self.viewer.status = "✅ Data loaded successfully."

    # 下方保留原有的事件处理函数
    def on_point_size_change(self, val):
        for layer in self.viewer.layers:
            if layer.name.startswith("Cell:") or layer.name == ">> Highlight Cells <<":
                layer.size = val

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
            sample_name = text.split(": ")[1]
            if "[Native]" in text:
                self.mode = "Native"
                self.load_sample_native_view(sample_name)
            elif "[Atlas ]" in text:
                self.mode = "Atlas_Sample"
                self.load_sample_atlas_view(sample_name)
    
    def on_class_single_change(self, text):
        self.current_class = text
        if text == "Volume": self.combo_metric.setCurrentText("Volume")
        elif self.combo_metric.currentText() == "Volume": self.combo_metric.setCurrentText("Count")
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