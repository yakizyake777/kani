import cv2
import numpy as np
import trimesh
from rembg import remove

# ==========================================
# 1. GUIクラス: 辺と面の指定用
# ==========================================
class SelectionTool:
    def __init__(self, image_path, window_name, mode="edge"):
        """
        mode="edge": 2点クリックで線分を指定
        mode="face": 複数点クリックで多角形面を指定
        """
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"{image_path} が見つかりません")
        
        self.mode = mode
        self.window_name = window_name
        self.points = []
        self.original_points = []
        
        # 表示用リサイズ
        self.scale = 1.0
        h, w = self.image.shape[:2]
        if h > 800 or w > 800:
            self.scale = 800 / max(h, w)
        self.display_image = cv2.resize(self.image, None, fx=self.scale, fy=self.scale)
        self.clean_image = self.display_image.copy()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 座標保存
            orig_x = int(x / self.scale)
            orig_y = int(y / self.scale)
            self.points.append((x, y))
            self.original_points.append((orig_x, orig_y))
            
            # 描画
            cv2.circle(self.display_image, (x, y), 4, (0, 0, 255), -1)
            
            if self.mode == "edge" and len(self.points) == 2:
                cv2.line(self.display_image, self.points[0], self.points[1], (0, 255, 0), 2)
            
            elif self.mode == "face" and len(self.points) > 1:
                cv2.line(self.display_image, self.points[-2], self.points[-1], (0, 255, 255), 1)

            cv2.imshow(self.window_name, self.display_image)

    def run(self):
        print(f"--- {self.window_name} ---")
        if self.mode == "edge":
            print("【辺選択】結合したい辺の「始点」と「終点」をクリックしてください。")
        else:
            print("【面選択】結合したい面を囲むように点をクリックしてください（3点以上）。")
        print("決定したら [Enter] キーを押してください。")

        cv2.imshow(self.window_name, self.display_image)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        while True:
            key = cv2.waitKey(0)
            if key == 13: # Enter
                break
        
        cv2.destroyWindow(self.window_name)
        return self.original_points

# ==========================================
# 2. メッシュ生成と座標取得
# ==========================================
def process_mesh(image_path, thickness_scale=50.0):
    """画像から3Dメッシュと深度マップ、グリッド情報を生成して返す"""
    img = cv2.imread(image_path)
    
    # 背景削除
    result = remove(img)
    mask = result[:, :, 3]
    img_rgb = cv2.cvtColor(result, cv2.COLOR_RGBA2RGB)

    # 深度マップ (Balloon Inflation)
    dist_img = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_dist = np.max(dist_img) if np.max(dist_img) > 0 else 1
    normalized_depth = dist_img / max_dist
    z_values = np.sin(normalized_depth * (np.pi / 2)) * thickness_scale

    # グリッド生成
    h, w = mask.shape
    x_grid, y_grid = np.meshgrid(np.arange(w) - w/2, np.arange(h) - h/2)
    y_grid = -y_grid 

    # 頂点データ
    xs, ys, zs = x_grid.flatten(), y_grid.flatten(), z_values.flatten()
    vertices = np.column_stack((xs, ys, zs))
    
    colors_rgb = img_rgb.reshape(-1, 3)
    colors_alpha = mask.reshape(-1, 1)
    vertex_colors = np.hstack((colors_rgb, colors_alpha)).astype(np.uint8)

    # 面データ (簡易生成)
    indices = np.arange(h * w).reshape(h, w)
    mask_flat = mask.flatten() > 10
    
    tl = indices[:-1, :-1].flatten()
    tr = indices[:-1, 1:].flatten()
    bl = indices[1:, :-1].flatten()
    br = indices[1:, 1:].flatten()
    
    f1 = np.column_stack((tl, bl, tr))
    f2 = np.column_stack((tr, bl, br))
    all_faces = np.vstack((f1, f2))
    valid_faces = mask_flat[all_faces].all(axis=1)
    faces = all_faces[valid_faces]

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors)
    
    # データ参照用辞書
    data = {
        "x_grid": x_grid, "y_grid": y_grid, "z_values": z_values,
        "width": w, "height": h, "mask": mask
    }
    return mesh, data

def get_3d_point(data, x, y):
    """ピクセル座標 (x, y) に対応する3D座標を取得"""
    w, h = data["width"], data["height"]
    x, y = int(np.clip(x, 0, w-1)), int(np.clip(y, 0, h-1))
    
    px = data["x_grid"][y, x]
    py = data["y_grid"][y, x]
    pz = data["z_values"][y, x]
    return np.array([px, py, pz])

# ==========================================
# 3. 結合処理
# ==========================================
def main():
    file_a = "image_front.jpg"
    file_b = "image_back.jpg"
    output_file = "face_connected_model.glb"
    thickness = 50.0

    try:
        # 1. 選択ツールの起動
        tool_a = SelectionTool(file_a, "Image A (Select EDGE)", mode="edge")
        pts_a = tool_a.run() # 2点
        
        tool_b = SelectionTool(file_b, "Image B (Select FACE)", mode="face")
        pts_b = tool_b.run() # 多角形

        if len(pts_a) != 2 or len(pts_b) < 3:
            print("選択がキャンセルされたか、点数が不足しています。")
            return

        print("モデル生成中...")
        mesh_a, data_a = process_mesh(file_a, thickness)
        mesh_b, data_b = process_mesh(file_b, thickness)

        # 2. モデルBの初期配置 (対角視点なので反転)
        mesh_b.apply_transform(trimesh.transformations.scale_matrix(-1.0, [0,0,0]))
        # ※反転したので、データBから取得する座標も後で反転させる必要がある

        # 3. ターゲット座標（辺の中心）の計算
        p_a1 = get_3d_point(data_a, pts_a[0][0], pts_a[0][1])
        p_a2 = get_3d_point(data_a, pts_a[1][0], pts_a[1][1])
        target_center = (p_a1 + p_a2) / 2.0

        # 4. ソース座標（面の中心）の計算
        # 指定された多角形内のピクセルをサンプリングするか、頂点の平均を使う
        # ここでは簡易的に「指定された頂点群の3D平均」を面の中心とします
        b_points_3d = []
        for (bx, by) in pts_b:
            pt = get_3d_point(data_b, bx, by)
            b_points_3d.append(pt)
        
        # モデルBは反転しているので、取得した生座標も反転(-1倍)させる
        b_points_3d = -np.array(b_points_3d)
        
        source_center = np.mean(b_points_3d, axis=0)

        # 5. 移動 (Translation)
        # 面の中心(source)を 辺の中心(target) に合わせるベクトル
        translation = target_center - source_center
        
        matrix = np.eye(4)
        matrix[:3, 3] = translation
        mesh_b.apply_transform(matrix)

        # 6. 保存
        combined = trimesh.util.concatenate([mesh_a, mesh_b])
        combined.export(output_file, file_type='glb')
        
        print(f"完了！ {output_file} に保存しました。")
        print("指定された面が、辺の位置に来るように結合されました。")

    except Exception as e:
        print(f"エラー: {e}")

if __name__ == "__main__":
    main()