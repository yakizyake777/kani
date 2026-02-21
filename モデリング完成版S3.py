import numpy as np
import cv2
import trimesh
import boto3
import json
from skimage import measure
from rembg import remove
from PIL import Image
from botocore.exceptions import NoCredentialsError

# Webサーバー(API)化するためのライブラリ
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
# React(別ドメイン)からの通信を許可する設定
CORS(app)

class Interactive_VisualHull:
    def __init__(self, resolution=256):
        self.res = resolution

    def remove_background(self, img_cv):
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

    def rotate_image_and_mask(self, img, mask, angle):
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

    # ★変更点：GUIで座標を取るのではなく、Reactから受け取った points を使うように変更
    def align_and_process(self, img_cv, points):
        mask_obj, img_masked = self.remove_background(img_cv)
        
        # Reactから受け取った座標 [ [x1, y1], [x2, y2] ] をタプルに変換
        p1 = (int(points[0][0]), int(points[0][1]))
        p2 = (int(points[1][0]), int(points[1][1]))

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        distance = np.sqrt(dx**2 + dy**2)

        theta = np.degrees(np.arctan2(dy, dx))
        rot_angle = 90 - theta

        rotated_img, rotated_mask = self.rotate_image_and_mask(img_masked, mask_obj, rot_angle)

        M_obj = cv2.moments(rotated_mask)
        if M_obj["m00"] == 0:
            raise ValueError("回転後画像から物体が見つかりません。")
            
        center_x = int(M_obj["m10"] / M_obj["m00"])
        center_y = int(M_obj["m01"] / M_obj["m00"])

        return rotated_img, rotated_mask, (center_x, center_y), distance

    # ★変更点：ファイルパスではなく、画像データ(配列)と座標データを直接受け取る
    def create_mesh_api(self, img1_cv, img2_cv, pts1, pts2, output_filename="output.glb"):
        img1, mask1, center1, dist1 = self.align_and_process(img1_cv, pts1)
        img2, mask2, center2, dist2 = self.align_and_process(img2_cv, pts2)
        
        if dist2 < 1.0:
            raise ValueError("側面の指定した2点が近すぎます。")

        scale_factor = dist1 / dist2
        
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
            raise ValueError("ボクセル消失。2点の指定位置を見直してください。")

        voxels_padded = np.pad(voxels, pad_width=1, mode='constant', constant_values=False)

        verts, faces, normals, values = measure.marching_cubes(voxels_padded, 0.5)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        trimesh.repair.fix_normals(mesh)
        trimesh.smoothing.filter_taubin(mesh, iterations=15)

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


def upload_to_s3(local_file_path, bucket_name, s3_file_name):
    print(f"AWS S3へアップロードしています: {s3_file_name} ...")
    
    
    s3 = boto3.client('s3', 
                      region_name='ap-northeast-1')

    try:
        s3.upload_file(
            local_file_path, 
            bucket_name, 
            s3_file_name,
            ExtraArgs={'ContentType': 'model/gltf-binary'}
        )
        url = f"https://{bucket_name}.s3.amazonaws.com/{s3_file_name}"
        return url
    except Exception as e:
        print(f"アップロード失敗: {e}")
        return None


# ==============================================================
# Web API エンドポイントの設定
# ==============================================================
@app.route('/generate', methods=['POST'])
def generate_endpoint():
    try:
        # 1. Reactから送られた画像ファイルを受け取る
        file1 = request.files.get('image1')
        file2 = request.files.get('image2')
        if not file1 or not file2:
            return jsonify({"error": "画像が不足しています"}), 400

        # メモリ上で画像をOpenCV形式に変換
        img1_cv = cv2.imdecode(np.frombuffer(file1.read(), np.uint8), cv2.IMREAD_COLOR)
        img2_cv = cv2.imdecode(np.frombuffer(file2.read(), np.uint8), cv2.IMREAD_COLOR)

        # 2. Reactから送られた座標(JSON)を受け取ってPythonのリストに変換
        # React側から JSON.stringify されて送られてくる例: "[[100, 200], [100, 800]]"
        pts1 = json.loads(request.form.get('points1', '[]'))
        pts2 = json.loads(request.form.get('points2', '[]'))

        if len(pts1) != 2 or len(pts2) != 2:
            return jsonify({"error": "座標データが正しくありません"}), 400

        print(f"画像1座標: {pts1}, 画像2座標: {pts2}")

        # 3. モデリング処理の実行
        output_glb = "my_interactive_model.glb"
        converter = Interactive_VisualHull(resolution=256)
        converter.create_mesh_api(img1_cv, img2_cv, pts1, pts2, output_glb)

        # 4. S3へアップロード
        bucket_name = "my-3d-modeling-bucket" # ★あなたのS3バケット名
        s3_url = upload_to_s3(output_glb, bucket_name, "latest_model.glb")

        if s3_url:
            # ★成功したらReactへURLを返す
            return jsonify({
                "message": "成功",
                "glb_url": s3_url
            }), 200
        else:
            return jsonify({"error": "S3アップロードに失敗しました"}), 500

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Flaskサーバーの起動 (ポート5000)
    print("Webサーバーを起動します。ngrok等のURLと連携してください。")
    app.run(host='0.0.0.0', port=5000, debug=True)