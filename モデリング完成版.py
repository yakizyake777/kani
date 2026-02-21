import numpy as np
import cv2
import trimesh
from skimage import measure
from rembg import remove
from PIL import Image

class Interactive_VisualHull:
    def __init__(self, resolution=256):
        # 解像度を256に引き上げ、薄い物体の表現力を向上
        self.res = resolution

    def remove_background(self, img_cv):
        """
        AI(rembg)を使用して背景を削除し、マスクと背景透過画像を生成します。
        """
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        result_pil = remove(pil_img)
        result_np = np.array(result_pil)
        
        if result_np.shape[2] == 4:
            mask_obj = result_np[:, :, 3]
        else:
            mask_obj = cv2.cvtColor(result_np, cv2.COLOR_RGB2GRAY)

        _, mask_obj = cv2.threshold(mask_obj, 128, 255, cv2.THRESH_BINARY)
        img_masked = cv2.bitwise_and(img_cv, img_cv, mask=mask_obj)
        
        return mask_obj, img_masked

    def get_user_points(self, img_display, window_name="Select 2 Points"):
        """
        ユーザーが画像上で2点（上端・下端）をクリックして取得するGUI。
        画像下部に専用のUI領域（コントロールバー）を設けてボタンの被りを防止。
        """
        points = []
        
        h, w = img_display.shape[:2]
        max_dim = 800
        scale = 1.0
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            display_scaled = cv2.resize(img_display, (int(w * scale), int(h * scale)))
        else:
            display_scaled = img_display.copy()

        sh, sw = display_scaled.shape[:2]
        ui_height = 60
        canvas_w = max(sw, 150) # ボタンが収まる最低幅を確保
        
        # UI領域の中央にボタンを配置
        btn_rect = (canvas_w // 2 - 50, sh + 10, 100, 40)

        def draw_ui(img_base):
            # 画像の下にUI用のグレーの領域を追加したキャンバスを作成
            canvas = np.full((sh + ui_height, canvas_w, 3), 50, dtype=np.uint8)
            canvas[0:sh, 0:sw] = img_base
            
            bx, by, bw, bh = btn_rect
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (200, 200, 200), -1)
            cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (50, 50, 50), 2)
            cv2.putText(canvas, "RESET", (bx + 15, by + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            for p in points:
                cv2.circle(canvas, p, 5, (0, 0, 255), -1)
            if len(points) == 2:
                cv2.line(canvas, points[0], points[1], (0, 255, 0), 2)
                
            return canvas

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                bx, by, bw, bh = btn_rect
                # ボタン領域のクリック判定
                if bx <= x <= bx + bw and by <= y <= by + bh:
                    points.clear()
                    print("点をリセットしました。")
                # 画像領域内のクリック判定のみ受け付ける
                elif 0 <= x < sw and 0 <= y < sh:
                    if len(points) < 2:
                        points.append((x, y))
                
                cv2.imshow(window_name, draw_ui(display_scaled))

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_callback)

        print(f"\n--- {window_name} ---")
        print("物体の「上」となる基準点、次に「下」となる基準点の順に2ヶ所クリックしてください。")
        print("・画面下部の「RESET」ボタンをクリックするか、[r]キーを押すと選び直せます。")
        print("・2点を選んだ後、[Enter]キーを押すと決定して次へ進みます。")

        cv2.imshow(window_name, draw_ui(display_scaled))
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter
                if len(points) == 2:
                    break
                else:
                    print("2点を選択してからEnterキーを押してください。")
            elif key == ord('r'):  # rキー
                points.clear()
                cv2.imshow(window_name, draw_ui(display_scaled))
                print("点をリセットしました。")
            elif key == 27:  # ESC
                cv2.destroyAllWindows()
                raise ValueError("キャンセルされました。")

        cv2.destroyAllWindows()

        p1 = (int(points[0][0] / scale), int(points[0][1] / scale))
        p2 = (int(points[1][0] / scale), int(points[1][1] / scale))

        return p1, p2

    def rotate_image_and_mask(self, img, mask, angle):
        """
        はみ出しを防ぎながら画像とマスクを回転させます。
        """
        h, w = img.shape[:2]
        cX, cY = (w // 2, h // 2)

        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        rotated_img = cv2.warpAffine(img, M, (nW, nH), borderValue=(255, 255, 255))
        rotated_mask = cv2.warpAffine(mask, M, (nW, nH), borderValue=0)
        
        return rotated_img, rotated_mask

    def align_and_process(self, image_path, window_name, debug_filename=None):
        print(f"[{image_path}] 読み込み・AI背景削除中...")
        img_cv = cv2.imread(image_path)
        if img_cv is None:
            raise FileNotFoundError(f"{image_path} が見つかりません。")

        mask_obj, img_masked = self.remove_background(img_cv)
        p1, p2 = self.get_user_points(img_masked, window_name)

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        distance = np.sqrt(dx**2 + dy**2)

        theta = np.degrees(np.arctan2(dy, dx))
        rot_angle = 90 - theta

        print(f"[{image_path}] 垂直になるよう {rot_angle:.2f} 度回転させます...")
        rotated_img, rotated_mask = self.rotate_image_and_mask(img_masked, mask_obj, rot_angle)

        M_obj = cv2.moments(rotated_mask)
        if M_obj["m00"] == 0:
            raise ValueError(f"{image_path} の回転後画像から物体が見つかりません。")
            
        center_x = int(M_obj["m10"] / M_obj["m00"])
        center_y = int(M_obj["m01"] / M_obj["m00"])

        if debug_filename:
            debug_img = rotated_img.copy()
            contours, _ = cv2.findContours(rotated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 2)
            cv2.circle(debug_img, (center_x, center_y), 8, (255, 0, 0), -1)
            cv2.imwrite(debug_filename, debug_img)
            print(f"  -> デバッグ画像保存: {debug_filename}")

        return rotated_img, rotated_mask, (center_x, center_y), distance

    def create_mesh(self, path_front, path_left, output_filename="output.glb"):
        img1, mask1, center1, dist1 = self.align_and_process(path_front, "Front Image", "debug_front.jpg")
        img2, mask2, center2, dist2 = self.align_and_process(path_left, "Left Image", "debug_left.jpg")
        
        print(f"\n正面の指定長さ: {dist1:.2f} px, 側面の指定長さ: {dist2:.2f} px")

        if dist2 < 1.0:
            raise ValueError("側面の指定した2点が近すぎます。")

        scale_factor = dist1 / dist2
        print(f"側面の拡大倍率: {scale_factor:.4f}")

        print("3D構築中...")
        
        # ボクセル空間の見切れによる「平面化」を防ぐ動的スケーリング
        h1, w1 = mask1.shape
        h2, w2 = mask2.shape
        
        base_scale = dist1 / 2.0
        side_scale = dist2 / 2.0
        
        max_x = max(center1[0], w1 - center1[0]) / base_scale
        max_y1 = max(center1[1], h1 - center1[1]) / base_scale
        max_z = max(center2[0], w2 - center2[0]) / side_scale
        max_y2 = max(center2[1], h2 - center2[1]) / side_scale
        
        voxel_range = max(max_x, max_y1, max_z, max_y2) * 1.1
        norm_scale = base_scale
        
        print(f"ボクセル範囲を自動調整しました: ±{voxel_range:.2f}")

        x, y, z = np.mgrid[-voxel_range:voxel_range:self.res*1j, 
                           -voxel_range:voxel_range:self.res*1j, 
                           -voxel_range:voxel_range:self.res*1j]
        
        voxels = np.ones(x.shape, dtype=bool)
        
        u1 = (x * norm_scale) + center1[0]
        v1 = (y * -norm_scale) + center1[1]
        
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
            raise ValueError("ボクセル消失。位置合わせに失敗しました。2点の指定位置を見直してください。")

        voxels_padded = np.pad(voxels, pad_width=1, mode='constant', constant_values=False)

        verts, faces, normals, values = measure.marching_cubes(voxels_padded, 0.5)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        
        trimesh.repair.fix_normals(mesh)

        # 2視点特有の角張りを丸めるスムージング処理
        print("モデルの形状を丸めて自然にしています...")
        trimesh.smoothing.filter_taubin(mesh, iterations=15)

        # 画像の色を3Dモデルに投影（頂点カラーの計算）
        print("表面に画像の模様を焼き付け中...")

        vx = (verts[:, 0] - 1) / (self.res - 1) * (2 * voxel_range) - voxel_range
        vy = (verts[:, 1] - 1) / (self.res - 1) * (2 * voxel_range) - voxel_range
        vz = (verts[:, 2] - 1) / (self.res - 1) * (2 * voxel_range) - voxel_range

        uv1_u = (vx * norm_scale) + center1[0]
        uv1_v = (vy * -norm_scale) + center1[1]
        uv2_u = (vz * norm_scale / scale_factor) + center2[0]
        uv2_v = (vy * -norm_scale / scale_factor) + center2[1]

        uv1_u = np.clip(np.round(uv1_u).astype(int), 0, w1 - 1)
        uv1_v = np.clip(np.round(uv1_v).astype(int), 0, h1 - 1)
        uv2_u = np.clip(np.round(uv2_u).astype(int), 0, w2 - 1)
        uv2_v = np.clip(np.round(uv2_v).astype(int), 0, h2 - 1)

        colors_front = img1[uv1_v, uv1_u][:, ::-1]
        colors_left  = img2[uv2_v, uv2_u][:, ::-1]

        mesh_normals = mesh.vertex_normals
        weight_z = np.abs(mesh_normals[:, 2]) 
        weight_x = np.abs(mesh_normals[:, 0]) 
        
        use_front = weight_z > weight_x
        final_colors = np.where(use_front[:, None], colors_front, colors_left)

        alpha = np.full((len(final_colors), 1), 255, dtype=np.uint8)
        mesh.visual.vertex_colors = np.hstack((final_colors, alpha))
        
        mesh.apply_translation(-mesh.centroid)
        mesh.export(output_filename)
        print(f"成功: {output_filename} を出力しました。Windowsの3DビューアーやBlender等で色付きで確認できます。")

if __name__ == "__main__":
    FRONT_IMG = "front.jpg"
    LEFT_IMG = "left.jpg"
    
    try:
        converter = Interactive_VisualHull(resolution=256)
        converter.create_mesh(FRONT_IMG, LEFT_IMG, "my_interactive_model.glb")
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
