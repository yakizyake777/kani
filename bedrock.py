import os
import uuid
import json
import base64
import numpy as np
import cv2
import trimesh
import boto3
from skimage import measure
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import traceback

app = FastAPI()

# Reactからの通信を許可するCORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("models", exist_ok=True)
app.mount("/models", StaticFiles(directory="models"), name="models")

def upload_to_s3(local_file_path, bucket_name, s3_file_name, region_name="us-east-1"):
    """
    完成した3DモデルをAWS S3にアップロードする関数
    """
    print(f"AWS S3へアップロードしています: {s3_file_name} ...")
    s3 = boto3.client('s3', region_name=region_name)

    try:
        s3.upload_file(
            local_file_path, 
            bucket_name, 
            s3_file_name,
            ExtraArgs={'ContentType': 'model/gltf-binary'}
        )
        # S3の公開URLを生成して返す
        url = f"https://{bucket_name}.s3.amazonaws.com/{s3_file_name}"
        print(f"アップロード成功！ URL: {url}")
        return url
    except Exception as e:
        print(f"S3アップロード失敗: {e}")
        return None

class Interactive_VisualHull_API:
    def __init__(self, resolution=256, aws_region="us-east-1"):
        self.res = resolution
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=aws_region)

    def remove_background(self, img_cv):
        """
        AWS Bedrock (Amazon Titan Image Generator) を使って背景を削除する
        """
        # --- Bedrockの制限(1408px)に合わせて、安全なサイズ(1400px)に縮小する ---
        max_size = 1400
        h, w = img_cv.shape[:2]
        if max(h, w) > max_size:
            scale = max_size / float(max(h, w))
            new_w, new_h = int(w * scale), int(h * scale)
            img_to_send = cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_to_send = img_cv

        # 画像をPNG形式のBase64エンコードに変換
        success, buffer = cv2.imencode('.png', img_to_send)
        if not success:
            raise ValueError("画像のエンコードに失敗しました。")
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        # Titan Image Generator V2 の背景切り抜きペイロード
        body = {
            "taskType": "BACKGROUND_REMOVAL",
            "backgroundRemovalParams": {
                "image": img_base64
            }
        }

        try:
            response = self.bedrock_runtime.invoke_model(
                modelId="amazon.titan-image-generator-v2:0",
                body=json.dumps(body),
                accept="application/json",
                contentType="application/json"
            )
            response_body = json.loads(response.get("body").read())
            
            out_base64 = response_body.get("images")[0]
            out_bytes = base64.b64decode(out_base64)
            out_np = np.frombuffer(out_bytes, np.uint8)
            
            result_img = cv2.imdecode(out_np, cv2.IMREAD_UNCHANGED)
            
            if result_img is not None and result_img.shape[2] == 4:
                mask_obj = result_img[:, :, 3]
            else:
                mask_obj = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)

            # API側で画像サイズが変更された場合に対処するため、元のサイズにリサイズして戻す
            if mask_obj.shape[:2] != (h, w):
                mask_obj = cv2.resize(mask_obj, (w, h), interpolation=cv2.INTER_NEAREST)

            _, mask_obj = cv2.threshold(mask_obj, 128, 255, cv2.THRESH_BINARY)
            img_masked = cv2.bitwise_and(img_cv, img_cv, mask=mask_obj)
            
            return mask_obj, img_masked

        except Exception as e:
            print(f"Bedrock API Error: {e}")
            traceback.print_exc()
            raise ValueError("AWS Bedrockでの背景切り抜きに失敗しました。")

    def rotate_image_and_mask(self, img, mask, angle):
        h, w = img.shape[:2]
        cX, cY = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY
        rotated_img = cv2.warpAffine(img, M, (nW, nH), borderValue=(255, 255, 255))
        rotated_mask = cv2.warpAffine(mask, M, (nW, nH), borderValue=0)
        return rotated_img, rotated_mask

    def align_and_process(self, image_path, points_json, align_type='vertical'):
        img_cv = cv2.imread(image_path)
        h, w = img_cv.shape[:2]

        points = json.loads(points_json)
        p1 = (int(points[0]['x'] * w), int(points[0]['y'] * h))
        p2 = (int(points[1]['x'] * w), int(points[1]['y'] * h))

        mask_obj, img_masked = self.remove_background(img_cv)

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        distance = np.sqrt(dx**2 + dy**2)
        theta = np.degrees(np.arctan2(dy, dx))

        if align_type == 'vertical':
            rot_angle = 90 - theta
        else:
            rot_angle = -theta

        rotated_img, rotated_mask = self.rotate_image_and_mask(img_masked, mask_obj, rot_angle)

        M_obj = cv2.moments(rotated_mask)
        center_x = int(M_obj["m10"] / M_obj["m00"]) if M_obj["m00"] > 0 else rotated_img.shape[1] // 2
        center_y = int(M_obj["m01"] / M_obj["m00"]) if M_obj["m00"] > 0 else rotated_img.shape[0] // 2

        return rotated_img, rotated_mask, (center_x, center_y), distance

    def process_front(self, image_path, points_json):
        img_cv = cv2.imread(image_path)
        h, w = img_cv.shape[:2]
        points = json.loads(points_json)
        
        p1 = (int(points[0]['x'] * w), int(points[0]['y'] * h))
        p2 = (int(points[1]['x'] * w), int(points[1]['y'] * h))
        p3 = (int(points[2]['x'] * w), int(points[2]['y'] * h))
        p4 = (int(points[3]['x'] * w), int(points[3]['y'] * h))

        mask_obj, img_masked = self.remove_background(img_cv)

        dx_y = p2[0] - p1[0]
        dy_y = p2[1] - p1[1]
        dist_y = np.sqrt(dx_y**2 + dy_y**2)
        theta = np.degrees(np.arctan2(dy_y, dx_y))
        rot_angle = 90 - theta

        rotated_img, rotated_mask = self.rotate_image_and_mask(img_masked, mask_obj, rot_angle)

        M_obj = cv2.moments(rotated_mask)
        center_x = int(M_obj["m10"] / M_obj["m00"]) if M_obj["m00"] > 0 else rotated_img.shape[1] // 2
        center_y = int(M_obj["m01"] / M_obj["m00"]) if M_obj["m00"] > 0 else rotated_img.shape[0] // 2

        dist_x = np.sqrt((p4[0]-p3[0])**2 + (p4[1]-p3[1])**2)

        return rotated_img, rotated_mask, (center_x, center_y), dist_y, dist_x

    def create_mesh(self, path_front, path_left, path_top, pts1_json, pts2_json, pts3_json, output_filename):
        img1, mask1, center1, dist1_y, dist1_x = self.process_front(path_front, pts1_json)
        img2, mask2, center2, dist2_y = self.align_and_process(path_left, pts2_json, align_type='vertical')
        img3, mask3, center3, dist3_x = self.align_and_process(path_top, pts3_json, align_type='horizontal')
        
        ratio_L = dist2_y / dist1_y  
        ratio_T = dist3_x / dist1_x  
        
        h1, w1 = mask1.shape
        h2, w2 = mask2.shape
        h3, w3 = mask3.shape
        
        voxel_range = max(w1, h1) * 0.6

        x, y, z = np.mgrid[-voxel_range:voxel_range:self.res*1j, 
                           -voxel_range:voxel_range:self.res*1j, 
                           -voxel_range:voxel_range:self.res*1j]
        voxels = np.ones(x.shape, dtype=bool)
        
        u1 = x + center1[0]
        v1 = -y + center1[1]

        u2 = -z * ratio_L + center2[0]
        v2 = -y * ratio_L + center2[1]

        u3 = x * ratio_T + center3[0]
        v3 = z * ratio_T + center3[1]

        def check_mask(mask, u, v):
            h, w = mask.shape
            u_idx, v_idx = u.astype(int), v.astype(int)
            valid = (u_idx >= 0) & (u_idx < w) & (v_idx >= 0) & (v_idx < h)
            res = np.zeros(u.shape, dtype=bool)
            res[valid] = mask[v_idx[valid], u_idx[valid]] > 0
            return res

        voxels &= check_mask(mask1, u1, v1)
        voxels &= check_mask(mask2, u2, v2)
        voxels &= check_mask(mask3, u3, v3)

        if not np.any(voxels):
            cv2.imwrite("debug_mask_front.jpg", mask1)
            cv2.imwrite("debug_mask_left.jpg", mask2)
            cv2.imwrite("debug_mask_top.jpg", mask3)
            raise ValueError("シルエットが重なる部分がありません。AIによる背景切り抜きに失敗しているか、撮影の向きが間違っています。")

        voxels_padded = np.pad(voxels, pad_width=1, mode='constant', constant_values=False)
        verts, faces, _, _ = measure.marching_cubes(voxels_padded, 0.5)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        trimesh.repair.fix_normals(mesh)
        trimesh.smoothing.filter_taubin(mesh, iterations=15)

        vx = (verts[:, 0] - 1) / (self.res - 1) * (2 * voxel_range) - voxel_range
        vy = (verts[:, 1] - 1) / (self.res - 1) * (2 * voxel_range) - voxel_range
        vz = (verts[:, 2] - 1) / (self.res - 1) * (2 * voxel_range) - voxel_range

        uv1_u = np.clip(np.round(vx + center1[0]).astype(int), 0, w1 - 1)
        uv1_v = np.clip(np.round(-vy + center1[1]).astype(int), 0, h1 - 1)
        
        uv2_u = np.clip(np.round(-vz * ratio_L + center2[0]).astype(int), 0, w2 - 1)
        uv2_v = np.clip(np.round(-vy * ratio_L + center2[1]).astype(int), 0, h2 - 1)
        
        uv3_u = np.clip(np.round(vx * ratio_T + center3[0]).astype(int), 0, w3 - 1)
        uv3_v = np.clip(np.round(vz * ratio_T + center3[1]).astype(int), 0, h3 - 1)

        colors_front = img1[uv1_v, uv1_u][:, ::-1]
        colors_left  = img2[uv2_v, uv2_u][:, ::-1]
        colors_top   = img3[uv3_v, uv3_u][:, ::-1]

        weight_z = np.abs(mesh.vertex_normals[:, 2])
        weight_x = np.abs(mesh.vertex_normals[:, 0])
        weight_y = np.abs(mesh.vertex_normals[:, 1])

        weights = np.column_stack((weight_z, weight_x, weight_y))
        best_view = np.argmax(weights, axis=1)

        final_colors = np.zeros_like(colors_front)
        final_colors[best_view == 0] = colors_front[best_view == 0]
        final_colors[best_view == 1] = colors_left[best_view == 1]
        final_colors[best_view == 2] = colors_top[best_view == 2]

        alpha = np.full((len(final_colors), 1), 255, dtype=np.uint8)
        mesh.visual.vertex_colors = np.hstack((final_colors, alpha))
        
        mesh.apply_translation(-mesh.centroid)
        mesh.export(output_filename)

@app.post("/generate")
async def generate_model(
    image1: UploadFile = File(...), 
    image2: UploadFile = File(...),
    image3: UploadFile = File(...),
    points1: str = Form(...),
    points2: str = Form(...),
    points3: str = Form(...)
):
    front_path = f"tmp_front_{uuid.uuid4()}.jpg"
    left_path = f"tmp_left_{uuid.uuid4()}.jpg"
    top_path = f"tmp_top_{uuid.uuid4()}.jpg"
    glb_filename = f"model_{uuid.uuid4()}.glb"

    with open(front_path, "wb") as buffer:
        shutil.copyfileobj(image1.file, buffer)
    with open(left_path, "wb") as buffer:
        shutil.copyfileobj(image2.file, buffer)
    with open(top_path, "wb") as buffer:
        shutil.copyfileobj(image3.file, buffer)

    converter = Interactive_VisualHull_API(resolution=256, aws_region="us-east-1")
    output_filepath = os.path.join("models", glb_filename)
    
    # 1. 3Dモデルの生成
    converter.create_mesh(front_path, left_path, top_path, points1, points2, points3, output_filepath)
    
    # 2. S3へのアップロード
    # ⚠️ 以下の bucket_name を、先ほど作成したS3バケット名に書き換えてください！
    bucket_name = "my-3d-modeling-bucket" 
    
    s3_url = upload_to_s3(output_filepath, bucket_name, glb_filename, region_name="us-east-1")

    # 3. 使い終わった一時画像ファイルとローカルの3Dモデルを削除して綺麗にする
    os.remove(front_path)
    os.remove(left_path)
    os.remove(top_path)
    if os.path.exists(output_filepath):
        os.remove(output_filepath) 

    if s3_url:
        return {"glb_url": s3_url}
    else:
        raise HTTPException(status_code=500, detail="S3へのアップロードに失敗しました。S3の権限やバケット名を確認してください。")

    return {"glb_url": glb_url}
