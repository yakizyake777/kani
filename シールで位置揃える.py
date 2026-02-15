import numpy as np
import cv2
import trimesh
from skimage import measure
import os

class SimpleVisualHull:
    def __init__(self, resolution=128):
        # 解像度: 64は高速・粗い、128は標準、256は低速・細かい
        self.res = resolution
    
    def detect_features(self, image_path, debug_filename=None):
        """
        画像を読み込み、特徴を検出し、デバッグ画像を保存します。
        """
        print(f"処理中: {image_path}")
        # 画像読み込み
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"{image_path} が見つかりません。")
        
        h, w = img.shape[:2]
        debug_img = img.copy() # デバッグ描画用のコピー
        
        # --- 1. 赤いシールの検出 (HSV色空間) ---
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # 赤色の範囲定義 (照明条件によって調整が必要な場合があります)
        lower_red1 = np.array([0, 120, 70])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 70])
        upper_red2 = np.array([180, 255, 255])
        
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        
        # ノイズ除去
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

        # 赤い領域の重心を計算
        M_red = cv2.moments(mask_red)
        if M_red["m00"] == 0:
             # デバッグ画像を保存してエラー終了
            if debug_filename: cv2.imwrite(debug_filename, debug_img)
            raise ValueError(f"{image_path} から赤いシールが検出できませんでした。デバッグ画像を確認してください。")
        red_x = int(M_red["m10"] / M_red["m00"])
        red_y = int(M_red["m01"] / M_red["m00"])
        
        # [デバッグ描画] 赤いシール位置 (赤色で表示)
        cv2.circle(debug_img, (red_x, red_y), 10, (0, 0, 255), 2)
        cv2.drawMarker(debug_img, (red_x, red_y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        cv2.putText(debug_img, "Red Seal", (red_x+10, red_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)


        # --- 2. 物体のシルエット検出 ---
        # グレースケール化
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 大津の二値化で背景と物体を分離 (背景が明るく、物体が暗いと仮定)
        # ※もし背景が黒で物体が明るい場合は cv2.THRESH_BINARY_INV を cv2.THRESH_BINARY に変更してください。
        _, mask_obj = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # ノイズ除去 (小さな穴を埋める)
        kernel_close = np.ones((7,7), np.uint8)
        mask_obj = cv2.morphologyEx(mask_obj, cv2.MORPH_CLOSE, kernel_close)
        # ノイズ除去 (小さなごみを取る)
        kernel_open = np.ones((5,5), np.uint8)
        mask_obj = cv2.morphologyEx(mask_obj, cv2.MORPH_OPEN, kernel_open)

        # 最大の輪郭のみを残す（背景のノイズ対策）
        contours, _ = cv2.findContours(mask_obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
             if debug_filename: cv2.imwrite(debug_filename, debug_img)
             raise ValueError(f"{image_path} から物体が検出できませんでした。背景が複雑すぎる可能性があります。")
        
        max_contour = max(contours, key=cv2.contourArea)
        mask_obj_clean = np.zeros_like(mask_obj)
        cv2.drawContours(mask_obj_clean, [max_contour], -1, 255, thickness=cv2.FILLED)
        mask_obj = mask_obj_clean # クリーンなマスクに入れ替え

        # 物体の重心を計算
        M_obj = cv2.moments(mask_obj)
        if M_obj["m00"] == 0:
            if debug_filename: cv2.imwrite(debug_filename, debug_img)
            raise ValueError(f"{image_path} の重心計算に失敗しました。")
        center_x = int(M_obj["m10"] / M_obj["m00"])
        center_y = int(M_obj["m01"] / M_obj["m00"])
        
        # [デバッグ描画] シルエットの輪郭 (黄色) と 重心 (青色)
        cv2.drawContours(debug_img, [max_contour], -1, (0, 255, 255), 2)
        cv2.circle(debug_img, (center_x, center_y), 8, (255, 0, 0), -1)
        cv2.putText(debug_img, "Center", (center_x+10, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # デバッグ画像の保存
        if debug_filename:
            cv2.imwrite(debug_filename, debug_img)
            print(f"  -> デバッグ画像を保存しました: {debug_filename}")

        return mask_obj, (center_x, center_y), (red_x, red_y), (w, h)

    def create_mesh(self, path_front, path_left, output_filename="output.glb"):
        print("--- 画像解析開始 ---")
        
        # デバッグ画像のファイル名を指定して解析を実行
        mask1, center1, red1, size1 = self.detect_features(path_front, debug_filename="debug_front.jpg")
        mask2, center2, red2, size2 = self.detect_features(path_left, debug_filename="debug_left.jpg")
        
        print("\n--- パラメータ計算 ---")
        print(f"正面(Front): 重心={center1}, 赤シール={red1}, Y距離={abs(red1[1] - center1[1])}")
        print(f"側面(Left) : 重心={center2}, 赤シール={red2}, Y距離={abs(red2[1] - center2[1])}")

        # --- 拡大縮小と平行移動のパラメータ計算 ---
        dist_y1 = abs(red1[1] - center1[1])
        dist_y2 = abs(red2[1] - center2[1])
        
        if dist_y2 < 5: # 距離が近すぎると計算が不安定になるためエラーにする
            raise ValueError("左画像の「重心」と「赤シール」のY座標が近すぎます。うまく検出できていないか、位置が悪いです。debug_left.jpgを確認してください。")

        # スケールファクター計算 (画像2を画像1の基準に合わせるための倍率)
        scale_factor = dist_y1 / dist_y2
        print(f"計算された拡大率 (Scale Factor): {scale_factor:.4f}")

        # --- 視体積交差法 (Voxel Carving) ---
        print("\n--- 3Dモデル生成開始 ---")
        print(f"ボクセル解像度: {self.res}^3")
        
        voxel_range = 1.2
        x, y, z = np.mgrid[-voxel_range:voxel_range:self.res*1j, 
                           -voxel_range:voxel_range:self.res*1j, 
                           -voxel_range:voxel_range:self.res*1j]
        
        voxels = np.ones(x.shape, dtype=bool)
        
        h1, w1 = mask1.shape
        h2, w2 = mask2.shape
        
        # 正規化スケール (画像短辺の半分を基準1.0とする)
        norm_scale = min(h1, w1) / 2.0
        
        # 画像1 (正面) への投影
        u1 = (x * norm_scale) + center1[0]
        v1 = (y * -norm_scale) + center1[1]
        
        # 画像2 (左側面) への投影 (補正適用)
        # Z軸が横方向(u2)に見える。Y軸(v2)はスケール補正と位置補正を逆算して適用。
        u2 = (z * norm_scale / scale_factor) + center2[0]
        v2 = (y * -norm_scale / scale_factor) + center2[1]

        # インデックス計算と範囲チェック関数の定義
        def get_mask_value(mask, u_float, v_float):
            h, w = mask.shape
            u, v = u_float.astype(int), v_float.astype(int)
            # 画像範囲外のピクセルを参照しないようにする
            valid_mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
            result = np.zeros(u.shape, dtype=bool)
            # 範囲内の場所だけマスクの値を参照
            result[valid_mask] = mask[v[valid_mask], u[valid_mask]] > 0
            return result

        # 交差判定 (Carving)
        voxels &= get_mask_value(mask1, u1, v1)
        voxels &= get_mask_value(mask2, u2, v2)

        # --- メッシュ生成 ---
        print("メッシュ構築中...")
        if not np.any(voxels):
            raise ValueError("生成された3Dモデルが空っぽです。検出がうまくいっていない可能性が高いです。デバッグ画像を確認してください。")

        verts, faces, normals, values = measure.marching_cubes(voxels, 0.5)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        # 重心を原点に移動
        mesh.apply_translation(-mesh.centroid)

        # エクスポート
        mesh.export(output_filename)
        print(f"\n完了! 出力ファイル: {output_filename}")
        print("※生成された 'debug_front.jpg' と 'debug_left.jpg' を確認して、意図通りに検出できているか見てください。")

# --- 実行部分 ---
if __name__ == "__main__":
    # 必要なライブラリ: pip install numpy opencv-python trimesh scikit-image
    
    # 画像ファイル名を指定
    FRONT_IMG = "front.jpg"
    LEFT_IMG = "left.jpg"
    OUTPUT_GLB = "my_model.glb"
    
    # 解像度設定 (例: テスト用=64, 本番用=128 or 256)
    resolution = 128

    converter = SimpleVisualHull(resolution=resolution)
    
    try:
        # 実行フォルダに画像があるか確認
        if not os.path.exists(FRONT_IMG) or not os.path.exists(LEFT_IMG):
             print(f"エラー: '{FRONT_IMG}' または '{LEFT_IMG}' が見つかりません。")
             print("スクリプトと同じフォルダに画像を置いてください。")
        else:
             converter.create_mesh(FRONT_IMG, LEFT_IMG, OUTPUT_GLB)
    except Exception as e:
        print(f"\n【エラーが発生しました】\n詳細: {e}")
        print("※ 'debug_front.jpg' などの画像が生成されていれば、それを手がかりに原因を特定してください。")
