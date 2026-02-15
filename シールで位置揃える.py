import numpy as np
import cv2
import trimesh
from skimage import measure
from rembg import remove
from PIL import Image
import io
import os

class AI_VisualHull:
    def __init__(self, resolution=128):
        self.res = resolution

    def detect_features_ai(self, image_path, debug_filename=None):
        """
        rembg (AI) を使用して高精度に背景を削除し、特徴点を検出します。
        """
        print(f"[{image_path}] AI解析を実行中...")
        
        # 1. 画像読み込み
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            raise FileNotFoundError(f"{image_path} が見つかりません。")
        
        h, w = img_cv.shape[:2]
        debug_img = img_cv.copy()

        # --- AIによる背景削除 (ここが変更点) ---
        # OpenCV画像をPIL形式に変換してAIに渡す
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # 背景削除実行 (alphaチャンネル付きの画像が返る)
        result_pil = remove(pil_img)
        
        # PIL -> OpenCV形式 (Alphaチャンネルを取り出してマスクにする)
        result_np = np.array(result_pil)
        
        # Alphaチャンネル (4番目の要素) がマスクになる
        # 0=透明(背景), 255=不透明(物体)
        if result_np.shape[2] == 4:
            mask_obj = result_np[:, :, 3]
        else:
            # 失敗時は従来の方法へフォールバック(念のため)
            print("警告: Alphaチャンネルが取得できませんでした。")
            mask_obj = cv2.cvtColor(result_np, cv2.COLOR_RGB2GRAY)

        # マスクの二値化保証 (AIが出す半透明ピクセルを0か255にはっきり分ける)
        _, mask_obj = cv2.threshold(mask_obj, 128, 255, cv2.THRESH_BINARY)

        # 物体の重心計算
        M_obj = cv2.moments(mask_obj)
        if M_obj["m00"] == 0:
            if debug_filename: cv2.imwrite(debug_filename, debug_img)
            raise ValueError(f"{image_path} から物体が見つかりません。")
            
        center_x = int(M_obj["m10"] / M_obj["m00"])
        center_y = int(M_obj["m01"] / M_obj["m00"])

        # --- 赤いシールの検出 ---
        # 背景削除された画像を使って赤検出を行うと、背景の赤色誤検出を防げます
        # 背景を黒で塗りつぶした画像を作成
        img_masked = cv2.bitwise_and(img_cv, img_cv, mask=mask_obj)
        hsv = cv2.cvtColor(img_masked, cv2.COLOR_BGR2HSV)
        
        # 赤色の範囲 (環境により調整)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
        
        # ノイズ除去
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))
        
        M_red = cv2.moments(mask_red)
        if M_red["m00"] == 0:
             # 赤が見つからない場合、デバッグ画像を保存
            cv2.drawContours(debug_img, [cv2.findContours(mask_obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]], -1, (0,255,0), 2)
            if debug_filename: cv2.imwrite(debug_filename, debug_img)
            raise ValueError(f"{image_path} から赤いシールが見つかりません。")
            
        red_x = int(M_red["m10"] / M_red["m00"])
        red_y = int(M_red["m01"] / M_red["m00"])

        # --- デバッグ画像の作成 ---
        # 検出した輪郭を緑で描画
        contours, _ = cv2.findContours(mask_obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2) # 緑の線
        
        # 重心を青丸
        cv2.circle(debug_img, (center_x, center_y), 8, (255, 0, 0), -1)
        
        # 赤シールを赤十字
        cv2.drawMarker(debug_img, (red_x, red_y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

        if debug_filename:
            cv2.imwrite(debug_filename, debug_img)
            print(f"  -> デバッグ画像保存: {debug_filename}")

        return mask_obj, (center_x, center_y), (red_x, red_y), (w, h)

    def create_mesh(self, path_front, path_left, output_filename="output.glb"):
        # 解析実行 (デバッグ画像を出力)
        mask1, center1, red1, size1 = self.detect_features_ai(path_front, "ai_debug_front.jpg")
        mask2, center2, red2, size2 = self.detect_features_ai(path_left, "ai_debug_left.jpg")
        
        print(f"正面Y距離: {abs(red1[1] - center1[1])}, 側面Y距離: {abs(red2[1] - center2[1])}")

        # スケール計算
        dist_y1 = abs(red1[1] - center1[1])
        dist_y2 = abs(red2[1] - center2[1])
        
        if dist_y2 < 1.0:
            raise ValueError("側面画像の重心とシールの高さが近すぎます。")

        scale_factor = dist_y1 / dist_y2
        print(f"拡大倍率: {scale_factor:.4f}")

        # --- ボクセル生成 ---
        print("3D構築中...")
        voxel_range = 1.2
        x, y, z = np.mgrid[-voxel_range:voxel_range:self.res*1j, 
                           -voxel_range:voxel_range:self.res*1j, 
                           -voxel_range:voxel_range:self.res*1j]
        
        voxels = np.ones(x.shape, dtype=bool)
        h1, w1 = mask1.shape
        h2, w2 = mask2.shape
        norm_scale = min(h1, w1) / 2.0
        
        # 正面からの投影
        u1 = (x * norm_scale) + center1[0]
        v1 = (y * -norm_scale) + center1[1]
        
        # 側面からの投影 (拡大補正 + 平行移動)
        u2 = (z * norm_scale / scale_factor) + center2[0]
        v2 = (y * -norm_scale / scale_factor) + center2[1]

        def check_mask(mask, u, v):
            h, w = mask.shape
            u_idx = u.astype(int)
            v_idx = v.astype(int)
            valid = (u_idx >= 0) & (u_idx < w) & (v_idx >= 0) & (v_idx < h)
            res = np.zeros(u.shape, dtype=bool)
            res[valid] = mask[v_idx[valid], u_idx[valid]] > 0
            return res

        voxels &= check_mask(mask1, u1, v1)
        voxels &= check_mask(mask2, u2, v2)

        if not np.any(voxels):
            raise ValueError("ボクセル消失。位置合わせ失敗。")

        # メッシュ化
        verts, faces, normals, values = measure.marching_cubes(voxels, 0.5)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        mesh.apply_translation(-mesh.centroid)
        mesh.export(output_filename)
        print(f"成功: {output_filename}")

if __name__ == "__main__":
    FRONT_IMG = "front.jpg"
    LEFT_IMG = "left.jpg"
    
    # 実行
    try:
        converter = AI_VisualHull(resolution=128)
        converter.create_mesh(FRONT_IMG, LEFT_IMG, "my_ai_model.glb")
    except Exception as e:
        print(f"エラー: {e}")
