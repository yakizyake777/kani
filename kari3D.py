import cv2
import numpy as np
import trimesh
from rembg import remove

def create_mesh_from_image_view(image_path, is_back=False, thickness_scale=40.0):
    """
    画像から深度推定を行い、メッシュ(Trimeshオブジェクト)を生成して返す関数
    """
    # --- 1. 画像読み込みと前処理 ---
    print(f"Processing: {image_path} (Back side: {is_back})")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"画像が見つかりません: {image_path}")

    # 背景削除 (rembg)
    result = remove(img)
    mask = result[:, :, 3] # アルファチャンネル
    img_rgb = cv2.cvtColor(result, cv2.COLOR_RGBA2RGB)

    # --- 2. 深度マップの作成 (Balloon Inflation) ---
    # 距離変換
    dist_img = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    max_dist = np.max(dist_img)
    if max_dist == 0: max_dist = 1
    
    # 正規化と丸み付け (0.0 ~ 1.0) -> 厚み付け
    normalized_depth = dist_img / max_dist
    z_values = np.sin(normalized_depth * (np.pi / 2)) * thickness_scale

    # --- 3. 頂点(Vertices)と面(Faces)の生成 ---
    h, w = mask.shape
    # グリッド生成 (画像の中心を原点に)
    x_grid, y_grid = np.meshgrid(np.arange(w) - w/2, np.arange(h) - h/2)
    y_grid = -y_grid # 画像座標(Y下)を3D座標(Y上)に変換

    # 頂点配列の作成 (H*W, 3)
    # フラット化
    xs = x_grid.flatten()
    ys = y_grid.flatten()
    zs = z_values.flatten()
    
    vertices = np.column_stack((xs, ys, zs))
    
    # 色情報の取得 (0-1に正規化せず、0-255のままTrimeshに渡すのが一般的だが、ここでは正規化して渡すことも可能)
    # trimeshのvertex_colorsは (N, 4) のRGBA uint8 が確実
    colors_rgb = img_rgb.reshape(-1, 3)
    colors_alpha = mask.reshape(-1, 1)
    vertex_colors = np.hstack((colors_rgb, colors_alpha)).astype(np.uint8)

    # 面データ(Faces)の作成: グリッドの三角形分割
    # ピクセル(i, j) のインデックス k に対し、
    #  (k, k+1, k+w), (k+1, k+w+1, k+w) の2つの三角形を作る
    
    # グリッドのインデックス配列 (H, W)
    indices = np.arange(h * w).reshape(h, w)
    
    # 右下端を除く範囲で処理
    # top-left vertices of the squares
    tl = indices[:-1, :-1].flatten()
    tr = indices[:-1, 1:].flatten() # top-right
    bl = indices[1:, :-1].flatten() # bottom-left
    br = indices[1:, 1:].flatten()  # bottom-right

    # 三角形1 (tl, bl, tr)
    faces1 = np.column_stack((tl, bl, tr))
    # 三角形2 (tr, bl, br)
    faces2 = np.column_stack((tr, bl, br))
    
    faces = np.vstack((faces1, faces2))

    # --- 4. マスク外の不要な面を削除 ---
    # 三角形の3つの頂点がすべてマスク内(alpha > 0)である場合のみ残す
    # または深度(z)が0より大きい場合のみ残す
    
    # 各頂点の有効判定 (mask > 10)
    valid_mask_flat = (mask.flatten() > 10)
    
    # faceの各頂点がvalidかチェック
    # (F, 3) の boolean 配列を作り、行ごとの論理積(all)をとる
    valid_faces = valid_mask_flat[faces].all(axis=1)
    
    faces = faces[valid_faces]

    # --- 5. メッシュ生成と座標変換 ---
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors)
    
    # 回転処理 (対角視点補正)
    mesh = rotate_mesh_diagonal(mesh, is_opposite=is_back)
    
    return mesh

def rotate_mesh_diagonal(mesh, is_opposite=False):
    """
    Trimeshオブジェクトを受け取り、回転行列を適用して返す
    """
    # 1. Y軸周りに -45度
    theta_y = np.radians(-45)
    Ry = np.array([
        [np.cos(theta_y), 0, np.sin(theta_y), 0],
        [0, 1, 0, 0],
        [-np.sin(theta_y), 0, np.cos(theta_y), 0],
        [0, 0, 0, 1]
    ])
    
    # 2. X軸周りに 35.26度
    theta_x = np.radians(35.264)
    Rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta_x), -np.sin(theta_x), 0],
        [0, np.sin(theta_x), np.cos(theta_x), 0],
        [0, 0, 0, 1]
    ])
    
    matrix = Rx @ Ry
    mesh.apply_transform(matrix)
    
    if is_opposite:
        # 反対側の場合、全座標を反転 (原点対称)
        # 行列で書くならScaling(-1, -1, -1)
        scale_matrix = np.eye(4)
        scale_matrix[0,0] = -1
        scale_matrix[1,1] = -1
        scale_matrix[2,2] = -1
        mesh.apply_transform(scale_matrix)
        
        # 鏡像反転すると面の向き(法線)が裏返るので修正
        mesh.invert()

    return mesh

def main():
    # --- 設定 ---
    # ※ここに画像のファイル名を入れてください
    file_front = "image_front.jpg" 
    file_back = "image_back.jpg"
    output_file = "output_model.glb"
    
    thickness = 50.0 # オブジェクトの厚み

    try:
        # 1. 前面メッシュ作成
        mesh_front = create_mesh_from_image_view(file_front, is_back=False, thickness_scale=thickness)
        
        # 2. 背面メッシュ作成
        mesh_back = create_mesh_from_image_view(file_back, is_back=True, thickness_scale=thickness)

        # 3. メッシュの結合
        # TrimeshのScene機能を使うか、単純に足し合わせる
        # 単純結合（1つのオブジェクトにする）
        combined_mesh = trimesh.util.concatenate([mesh_front, mesh_back])

        # 4. GLBとしてエクスポート
        print(f"Exporting to {output_file}...")
        combined_mesh.export(output_file, file_type='glb')
        
        print("完了しました！")
        print(f"生成されたファイル: {output_file}")
        print("Windowsの「3Dビューアー」やWebブラウザで開いて確認してください。")

    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()