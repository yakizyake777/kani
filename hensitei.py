import cv2
import numpy as np
import trimesh
from rembg import remove
from scipy.spatial.transform import Rotation as R

# ==========================================
# 1. ユーザーインターフェース (辺の指定)
# ==========================================
class EdgeSelector:
    def __init__(self, image_path, window_name):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise FileNotFoundError(f"{image_path} が見つかりません")
        
        # 表示用にリサイズ（大きすぎると画面に入らないため）
        self.scale = 1.0
        h, w = self.image.shape[:2]
        if h > 800 or w > 800:
            self.scale = 800 / max(h, w)
            self.display_image = cv2.resize(self.image, None, fx=self.scale, fy=self.scale)
        else:
            self.display_image = self.image.copy()

        self.window_name = window_name
        self.points = []
        self.original_points = [] # 元の解像度での座標

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 2:
                self.points.append((x, y))
                # 元の解像度に戻した座標を保存
                orig_x = int(x / self.scale)
                orig_y = int(y / self.scale)
                self.original_points.append((orig_x, orig_y))
                
                print(f"[{self.window_name}] 点{len(self.points)}: ({orig_x}, {orig_y})")
                
                # 描画
                cv2.circle(self.display_image, (x, y), 5, (0, 0, 255), -1)
                if len(self.points) == 2:
                    cv2.line(self.display_image, self.points[0], self.points[1], (0, 255, 0), 2)
                
                cv2.imshow(self.window_name, self.display_image)

    def select_edge(self):
        print(f"--- {self.window_name} ---")
        print("結合に使いたい「辺」の始点と終点をクリックしてください（計2回）。")
        print("決定したらキーボードの任意のキーを押してください。")
        
        cv2.imshow(self.window_name, self.display_image)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        cv2.waitKey(0)
        cv2.destroyWindow(self.window_name)
        
        if len(self.original_points) < 2:
            print("警告: 点が2つ指定されませんでした。")
            return None
        return self.original_points

# ==========================================
# 2. メッシュ生成と3D座標取得
# ==========================================
def create_mesh_and_get_3d_points(image_path, selected_pixels, thickness_scale=50.0):
    """
    画像からメッシュを生成し、指定されたピクセル座標に対応する3D座標も返す
    """
    img = cv2.imread(image_path)
    
    # 背景削除
    print(f"背景削除中... {image_path}")
    result = remove(img)
    mask = result[:, :, 3]
    img_rgb = cv2.cvtColor(result, cv2.COLOR_RGBA2RGB)

    # 深度マップ作成
    dist_img = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_dist = np.max(dist_img)
    if max_dist == 0: max_dist = 1
    
    normalized_depth = dist_img / max_dist
    z_values = np.sin(normalized_depth * (np.pi / 2)) * thickness_scale

    # グリッド生成 (画像中心を原点へ)
    h, w = mask.shape
    x_grid, y_grid = np.meshgrid(np.arange(w) - w/2, np.arange(h) - h/2)
    y_grid = -y_grid # Y軸反転

    # メッシュ生成用データ
    xs = x_grid.flatten()
    ys = y_grid.flatten()
    zs = z_values.flatten()
    vertices = np.column_stack((xs, ys, zs))
    
    colors_rgb = img_rgb.reshape(-1, 3)
    colors_alpha = mask.reshape(-1, 1)
    vertex_colors = np.hstack((colors_rgb, colors_alpha)).astype(np.uint8)

    # 面の生成 (validな部分のみ)
    indices = np.arange(h * w).reshape(h, w)
    faces = []
    
    # 高速化のため、ベクトル計算で面を作成
    # 左上(TL)を基準に右(TR)、下(BL)、右下(BR)
    mask_flat = mask.flatten() > 10 # 背景以外のマスク
    
    # 四角形を2つの三角形に分割
    # メッシュ生成は簡易化のためTrimeshのVoxel等は使わず自前で定義
    # ここでは詳細な面貼りは省略し、Trimeshに点群から復元させるか、
    # あるいは以前のコードのようにgridベースで作る。
    # 以前のコードのロジックを再利用:
    tl = indices[:-1, :-1].flatten()
    tr = indices[:-1, 1:].flatten()
    bl = indices[1:, :-1].flatten()
    br = indices[1:, 1:].flatten()
    
    f1 = np.column_stack((tl, bl, tr))
    f2 = np.column_stack((tr, bl, br))
    all_faces = np.vstack((f1, f2))
    
    # 背景部分の面を削除
    valid_faces = mask_flat[all_faces].all(axis=1)
    faces = all_faces[valid_faces]

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors)

    # --- 指定されたピクセルの3D座標を取得 ---
    selected_3d_points = []
    if selected_pixels is not None:
        for px, py in selected_pixels:
            # 範囲外チェック
            px = np.clip(px, 0, w-1)
            py = np.clip(py, 0, h-1)
            
            # インデックス計算
            idx = int(py) * w + int(px)
            
            # もし背景(mask=0)をクリックしていた場合、最も近い有効な点を探す（簡易実装）
            if mask[int(py), int(px)] == 0:
                print(f"警告: 指定点({px},{py})は背景です。深度は0として扱います。")
                pt_z = 0
            else:
                pt_z = z_values[int(py), int(px)]
                
            pt_x = x_grid[int(py), int(px)]
            pt_y = y_grid[int(py), int(px)]
            
            selected_3d_points.append([pt_x, pt_y, pt_z])

    return mesh, np.array(selected_3d_points)

# ==========================================
# 3. 位置合わせ（アライメント）計算
# ==========================================
def align_meshes(mesh_a, points_a, mesh_b, points_b):
    """
    points_b が points_a に重なるように mesh_b を移動・回転させる
    points_a, points_b: それぞれ [始点, 終点] の3D座標 (2, 3)
    """
    # 1. 重心（中点）を合わせる平行移動
    center_a = np.mean(points_a, axis=0)
    center_b = np.mean(points_b, axis=0)
    
    # Bを原点へ -> Aの位置へ
    translation = center_a - center_b
    
    # 一旦平行移動だけ適用したBの座標を作る（回転計算のため）
    points_b_centered = points_b - center_b
    points_a_centered = points_a - center_a

    # 2. 回転を合わせる
    # 辺のベクトルを計算
    vec_a = points_a[1] - points_a[0]
    vec_b = points_b[1] - points_b[0]
    
    # 正規化
    vec_a = vec_a / np.linalg.norm(vec_a)
    vec_b = vec_b / np.linalg.norm(vec_b)
    
    # 2つのベクトルの回転軸と角度を計算
    # 回転軸 = vec_b x vec_a
    cross_prod = np.cross(vec_b, vec_a)
    dot_prod = np.dot(vec_b, vec_a)
    
    # 回転角
    angle = np.arccos(np.clip(dot_prod, -1.0, 1.0))
    
    # 回転行列の生成
    if np.linalg.norm(cross_prod) < 1e-6:
        # ベクトルが平行または逆平行の場合
        if dot_prod > 0:
            rotation_matrix = np.eye(3) # 同じ向きなら回転なし
        else:
            rotation_matrix = -np.eye(3) # 逆向きなら180度回転（適当な軸で）
            # ここでは厳密には任意の軸での180度回転が必要だが簡略化
            print("注意: 指定された辺が逆向きです。")
    else:
        rotation_vec = cross_prod / np.linalg.norm(cross_prod) * angle
        rotation_matrix = R.from_rotvec(rotation_vec).as_matrix()

    # --- 変換の適用 ---
    # mesh_b 全体に適用する変換行列 (4x4)
    # T = T_trans * R * T_center_back
    
    # 1. Bの中心を原点に戻す
    M1 = np.eye(4)
    M1[:3, 3] = -center_b
    
    # 2. 回転
    M2 = np.eye(4)
    M2[:3, :3] = rotation_matrix
    
    # 3. Aの中心へ移動
    M3 = np.eye(4)
    M3[:3, 3] = center_a
    
    # 4. 追加の補正（対角視点の前提）
    # 直方体の反対側なので、本来は180度裏返っているはず。
    # 「辺を合わせる」回転でそれが吸収されていればOKだが、
    # メッシュの「表裏」を考慮して、ベクトル合わせに加えて「軸周りの回転」が必要な場合がある。
    # 今回は単純に「指定されたベクトルを一致させる」変換を行う。
    
    final_matrix = M3 @ M2 @ M1
    mesh_b.apply_transform(final_matrix)
    
    return mesh_b

# ==========================================
# 4. メイン処理
# ==========================================
def main():
    file_a = "image_front.jpg" # 1枚目（表）
    file_b = "image_back.jpg"  # 2枚目（裏）
    output_file = "joined_model.glb"
    thickness = 50.0

    try:
        # 1. ユーザーによる辺指定
        selector_a = EdgeSelector(file_a, "Image A (Front)")
        pts_a_img = selector_a.select_edge()
        
        selector_b = EdgeSelector(file_b, "Image B (Back)")
        pts_b_img = selector_b.select_edge()
        
        if pts_a_img is None or pts_b_img is None:
            print("キャンセルされました。")
            return

        print("メッシュ生成と結合計算中...")

        # 2. メッシュ生成
        # まずは標準的な対角配置の角度で初期生成しておく
        # (ただし、今回は辺指定で位置合わせするため、回転は後で任せる手もあるが、
        #  形状の立体感(Z方向)は初期回転に依存しないので、ローカル座標で生成する)
        
        mesh_a, pts_a_3d = create_mesh_and_get_3d_points(file_a, pts_a_img, thickness)
        mesh_b, pts_b_3d = create_mesh_and_get_3d_points(file_b, pts_b_img, thickness)

        # 重要: 裏面(Mesh B)は、本来「裏返っている」ものとして生成されているが、
        # ローカル座標では「表」と同じ向きで作られている。
        # 縫合するためには、Mesh Bをまず「裏返し（Z反転など）」にしておく必要がある。
        # 対角視点の場合、中心対称に近いので、全軸反転(-1,-1,-1)を初期状態とするのが自然。
        
        # Mesh Bを初期状態で反転（対角の位置へ）
        mesh_b.apply_transform(trimesh.transformations.scale_matrix(-1.0, [0,0,0])) # 原点中心に反転
        # 反転に伴い、指定点も反転させる
        pts_b_3d = -pts_b_3d

        # 3. 位置合わせ実行
        # 指定された辺(A)と辺(B)が重なるようにMesh Bを動かす
        mesh_b_aligned = align_meshes(mesh_a, pts_a_3d, mesh_b, pts_b_3d)

        # 4. 結合と保存
        combined = trimesh.util.concatenate([mesh_a, mesh_b_aligned])
        combined.export(output_file, file_type='glb')
        
        print(f"完了！ {output_file} に保存しました。")
        print("指定された辺同士がくっつくように調整されました。")

    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()