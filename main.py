from fastapi import FastAPI, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import subprocess

app = FastAPI()

# CORS設定（フロントエンドからのアクセスを許可）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 保存場所
UPLOAD_DIR = "uploaded_images"
OUTPUT_DIR = "workspace/dense"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# COLMAPを実行する関数（さっきのコードの中身）
def run_colmap_task():
    print("COLMAP処理開始...")
    # ここにさっきのCOLMAP実行コマンドを書く
    # subprocess.run(["colmap", "automatic_reconstructor", ...])
    print("COLMAP処理完了！")

@app.post("/upload/")
async def upload_images(files: list[UploadFile], background_tasks: BackgroundTasks):
    # 1. 前回の画像を消して、新しい画像を保存
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR)

    for file in files:
        file_location = f"{UPLOAD_DIR}/{file.filename}"
        with open(file_location, "wb+") as file_object:
            file_object.write(file.file.read())
    
    # 2. 裏側でCOLMAPを動かす予約をする（これが重要！ユーザーを待たせない）
    background_tasks.add_task(run_colmap_task)
    
    return {"message": "アップロード完了。3D生成を開始しました。"}

@app.get("/model/")
def get_model():
    # 生成された .ply ファイルを返す
    ply_path = f"{OUTPUT_DIR}/meshed-poisson.ply"
    if os.path.exists(ply_path):
        return FileResponse(ply_path)
    return {"status": "processing", "message": "まだ生成中です"}
