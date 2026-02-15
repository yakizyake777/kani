import numpy as np
import cv2
import trimesh
from skimage import measure
import os

class SimpleVisualHull:
    def __init__(self, resolution=128):
        self.res = resolution  # ボクセルの解像度 (高いほど詳細だが重い)
    
    def detect_features(self, image_path):
        """
        画像を読み込み、以下の情報を返します:
        - mask: 物体のシルエット (0 or 255)
        - center: 重心座標 (x, y)
        - red_pos: 赤いシールの座標 (x, y)
        """
        # 画像読み込み
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"{image_path} が見つかりません。")
        
        h, w = img.shape[:2]
        
        # 1. 赤いシールの検出 (HSV色空間を使用)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 赤色はHueが0付近と180付近に分かれるため2回マスクを作成
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = mask_red1 + mask_red2
        
        # 赤い領域の重心を計算
        M_red = cv2.moments(mask_red)
        if M_red["m00"] == 0:
            raise ValueError(f"{image_path} から赤いシールが検出できませんでした。閾値を調整してください。")
        red_x = int(M_red["m10"] / M_red["m00"])
        red_y = int(M_red["m01"] / M_red["m00"])
        
        # 2. 物体のシルエット検出 (ここでは簡易的に「白背景以外」または「黒背景上の物体」を想定)
        # 状況に合わせて調整してください。以下は「明るい背景にある暗い物体」またはその逆を自動判別して2値化します。
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 大津の二値化
        _, mask_obj = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # ノイズ除去
        kernel = np.ones((5,5), np.uint8)
        mask_obj = cv2.morphologyEx(mask_obj, cv2.MORPH_OPEN, kernel)
        
        # 物体の重心を計算
        M_obj = cv2.moments(mask_obj)
        if M_obj["m00"] == 0:
            raise ValueError(f"{image_path} から物体が検出できませんでした。")
        center_x = int(M_obj["m10"] / M_obj["m00"])
        center_y = int(M_obj["m01"] / M_obj["m00"])
        
        return mask_obj, (center_x, center_y), (red_x, red_y), (w, h)

    def create_mesh(self, path_front, path_left, output_filename="output.glb"):
        print("画像を解析中...")
        
        # --- 手順① & ②: 画像情報の取得 ---
        mask1, center1, red1, size1 = self.detect_features(path_front)
        mask2, center2, red2, size2 = self.detect_features(path_left)
        
        print(f"正面画像 - 重心: {center1}, 赤シール: {red1}")
        print(f"左側画像 - 重心: {center2}, 赤シール: {red2}")

        # --- 手順③: 拡大縮小と平行移動のパラメータ計算 ---
        # 基準は「正面画像(1)」とする
        
        # 1. 重心と赤シールのY座標の距離を計算
        # 絶対値を取ります（シールが重心の上にあるか下にあるかの一貫性は必要）
        dist_y1 = abs(red1[1] - center1[1])
        dist_y2 = abs(red2[1] - center2[1])
        
        if dist_y2 == 0:
            raise ValueError("左画像の重心と赤シールのY座標が同じです。スケーリング計算ができません。")

        # 2. スケールファクター (Scale) の計算
        # 画像2の距離を画像1の距離に合わせるための倍率
        scale_factor = dist_y1 / dist_y2
        print(f"計算された拡大率: {scale_factor:.4f}")

        # --- 手順④: 視体積交差法 (Voxel Carving) ---
        print("ボクセル空間を生成中...")
        
        # ボクセルグリッドの作成 (-1.0 〜 1.0 の範囲)
        voxel_range = 1.2 # 余白を持たせる
        x, y, z = np.mgrid[-voxel_range:voxel_range:self.res*1j, 
                           -voxel_range:voxel_range:self.res*1j, 
                           -voxel_range:voxel_range:self.res*1j]
        
        voxels = np.ones(x.shape, dtype=bool)
        
        # 画像座標系への変換関数 (3D座標 -> 2Dピクセル)
        # 正面画像 (img1): 3Dの (X, Y) 平面に対応
        # 左側画像 (img2): 3Dの (Z, Y) 平面に対応 (左から撮影 = Z軸が横方向に見える)
        
        h1, w1 = mask1.shape
        h2, w2 = mask2.shape
        
        # --- 座標変換ロジック ---
        # ボクセル座標をピクセル座標に変換する際、手順③の補正を適用します。
        
        # 画像1 (正面) のマッピング: 単純なスケーリングとセンタリング
        # ボクセル(0,0) を 画像の重心(center1) に合わせる
        # 画像サイズに合わせて正規化係数を決める (画像の短い辺の半分を1.0とするなど)
        norm_scale = min(h1, w1) / 2.0
        
        u1 = (x * norm_scale) + center1[0]
        v1 = (y * -norm_scale) + center1[1] # 画像Y軸は下向きなので反転
        
        # 画像2 (左) のマッピング:
        # 3Dの Z軸 が 画像2の X軸(u2) に対応
        # 3Dの Y軸 が 画像2の Y軸(v2) に対応
        
        # ここで手順③の「拡大」と「平行移動」を適用します。
        # 変換式: v2_raw = (v2_corrected - shift) / scale
        # つまり、ボクセルのY座標を画像2のピクセルに直すとき、逆変換を行います。
        
        # 画像2における基準サイズ（スケール補正前）
        # 画像1と同じ物理スケール感で投影するため、norm_scaleを使用
        
        # X軸(横)方向: Z座標を使用。
        # 左から撮影しているので、3DのZが画像2の横軸(u)。
        # 画像2の中心Xに合わせて配置
        u2 = (z * norm_scale / scale_factor) + center2[0] 
        
        # Y軸(縦)方向:
        # center1[1] (画像1の重心Y) に対応する点が、画像2では center2[1] になるべき。
        # かつ、スケールは scale_factor で割る必要がある（画像2を拡大して合わせる＝座標を縮小して参照する）。
        
        # 理論:
        # Pixel_Y2_Corrected = (Pixel_Y2_Raw - Center2_Y) * Scale + Center1_Y
        # これが 画像1のPixel (v1) と等価になるようにボクセルを掘る。
        # 逆に、ボクセルYから画像2のピクセル(v2)を求めるとき:
        
        v2 = (y * -norm_scale / scale_factor) + center2[1]

        # --- 交差判定 (Carving) ---
        # 範囲外およびマスク外を False にする
        
        # 正面画像チェック
        valid_u1 = (u1 >= 0) & (u1 < w1)
        valid_v1 = (v1 >= 0) & (v1 < h1)
        # 整数のインデックスに変換
        u1_idx = u1.astype(int)
        v1_idx = v1.astype(int)
        
        # 画像範囲内かつマスクが黒(0)でない場所(=物体)のみTrue
        # ※ mask[y, x] でアクセス
        voxels[valid_u1 & valid_v1] &= (mask1[v1_idx[valid_u1 & valid_v1], u1_idx[valid_u1 & valid_v1]] > 0)
        
        # 左画像チェック
        valid_u2 = (u2 >= 0) & (u2 < w2)
        valid_v2 = (v2 >= 0) & (v2 < h2)
        u2_idx = u2.astype(int)
        v2_idx = v2.astype(int)
        
        voxels[valid_u2 & valid_v2] &= (mask2[v2_idx[valid_u2 & valid_v2], u2_idx[valid_u2 & valid_v2]] > 0)

        # 範囲外は削る
        voxels[~valid_u1 | ~valid_v1] = False
        voxels[~valid_u2 | ~valid_v2] = False

        # --- メッシュ生成 ---
        print("メッシュを構築中...")
        if not np.any(voxels):
            raise ValueError("ボクセルが全て削られてしまいました。位置合わせか閾値を確認してください。")

        # Marching Cubes法
        verts, faces, normals, values = measure.marching_cubes(voxels, 0.5)
        
        # 座標系を調整して見やすくする (センタリング等)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.apply_translation(-mesh.centroid)
        
        # 90度回転させて直立させるなどの調整 (必要に応じて)
        # mesh.apply_transform(trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0]))

        # エクスポート
        mesh.export(output_filename)
        print(f"完了: {output_filename} を作成しました。")

# --- 実行部分 ---
if __name__ == "__main__":
    # 画像ファイル名を指定してください
    FRONT_IMG = "front.jpg"  # ① 正面画像
    LEFT_IMG = "left.jpg"    # ② 左側画像
    
    converter = SimpleVisualHull(resolution=128) # 解像度は64, 128, 256など
    
    try:
        converter.create_mesh(FRONT_IMG, LEFT_IMG, "my_model.glb")
    except Exception as e:
        print(f"エラーが発生しました: {e}")