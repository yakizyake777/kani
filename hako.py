import numpy as np
import trimesh
from PIL import Image

def create_custom_mapped_box(
    front_img_path, 
    back_img_path, 
    face_mapping, 
    output_path="custom_box.glb", 
    thickness=0.2
):
    """
    2枚の画像から指定された領域を切り出し、箱の各面に貼り付ける関数
    
    Args:
        front_img_path: 正面画像のパス (source_id=0)
        back_img_path: 背面画像のパス (source_id=1)
        face_mapping: 面ごとの設定辞書 (下部main関数参照)
        thickness: 箱の厚み
    """
    
    # 1. 画像の読み込み
    images = {
        0: Image.open(front_img_path).convert("RGBA"),
        1: Image.open(back_img_path).convert("RGBA")
    }
    
    # 基準サイズ（正面画像のサイズを基準にする）
    base_w, base_h = images[0].size
    
    # 2. テクスチャアトラス（統合画像）の作成
    # 6つの面を横一列（または自由配置）に並べた大きな画像を作る
    # ここでは管理しやすいように、各面のテクスチャをリサイズして横に並べます
    
    # 各面のテクスチャ解像度（高画質用）
    tex_size = 1024 
    atlas_width = tex_size * 6
    atlas_height = tex_size
    
    atlas = Image.new("RGBA", (atlas_width, atlas_height), (0,0,0,0))
    
    # 面の順番定義 (Trimeshの頂点生成順と合わせる)
    # Front, Back, Top, Bottom, Right, Left
    face_order = ['front', 'back', 'top', 'bottom', 'right', 'left']
    
    # UV座標を格納するリスト
    uv_coords = []
    
    print("--- テクスチャ生成状況 ---")
    
    for i, face_name in enumerate(face_order):
        config = face_mapping.get(face_name)
        
        if config:
            source_id = config['source'] # 0 or 1
            # 領域指定 [x, y, w, h] (0.0~1.0の比率)
            rect = config['rect'] 
            
            src_img = images[source_id]
            W, H = src_img.size
            
            # ピクセル座標に変換
            crop_x = int(rect[0] * W)
            crop_y = int(rect[1] * H)
            crop_w = int(rect[2] * W)
            crop_h = int(rect[3] * H)
            
            # 切り抜き
            crop = src_img.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
            
            # アトラス用にリサイズして貼り付け
            # (箱へのマッピング時に引き伸ばされるため、ここで正方形にしてもOK)
            crop_resized = crop.resize((tex_size, tex_size), Image.Resampling.LANCZOS)
            atlas.paste(crop_resized, (i * tex_size, 0))
            
            print(f"[{face_name}] 画像{source_id}の ({rect}) を使用 -> Atlas位置: {i}")
            
        else:
            # 設定がない場合は黒塗り
            print(f"[{face_name}] 設定なし (黒)")
            draw = Image.new("RGBA", (tex_size, tex_size), (0, 0, 0, 255))
            atlas.paste(draw, (i * tex_size, 0))

        # UV座標の計算 (アトラス上の位置)
        # u_start, v_start, u_end, v_end
        u0 = (i * tex_size) / atlas_width
        u1 = ((i + 1) * tex_size) / atlas_width
        v0 = 0.0
        v1 = 1.0 # 横一列なので縦は全部使う
        
        # 4頂点分のUV (Counter-Clockwise: 左下, 右下, 右上, 左上 ではない。頂点順序に依存)
        # Trimeshの箱の面の頂点順序に合わせてUVを指定する
        # 通常は [左下, 右下, 右上, 左上] の順序でマッピングされることが多いが
        # ここでは頂点定義に合わせて調整する
        
        # 頂点定義順: 左下(-x,-y), 右下(x,-y), 右上(x,y), 左上(-x,y) と仮定して
        # Texture画像は左上が(0,0)なので、Vを反転させる必要がある場合がある
        uv_coords.append([u0, 1]) # Bottom-Left (画像の下)
        uv_coords.append([u1, 1]) # Bottom-Right
        uv_coords.append([u1, 0]) # Top-Right (画像の上)
        uv_coords.append([u0, 0]) # Top-Left
    
    uvs = np.array(uv_coords).reshape(-1, 2)

    # 3. 箱メッシュの作成
    # 画像のアスペクト比に合わせて箱のサイズを決める
    aspect_ratio = base_h / base_w
    box_w = 1.0
    box_h = aspect_ratio
    box_d = thickness

    x, y, z = box_w/2, box_h/2, box_d/2

    # 24頂点 (6面 * 4頂点) の定義
    # これにより各面で独立したUVを持てる（ハードエッジ）
    vertices = np.array([
        # Front (Z+)
        [-x, -y,  z], [ x, -y,  z], [ x,  y,  z], [-x,  y,  z],
        # Back (Z-) - 裏から見て正立するように頂点順を工夫
        [ x, -y, -z], [-x, -y, -z], [-x,  y, -z], [ x,  y, -z],
        # Top (Y+)
        [-x,  y,  z], [ x,  y,  z], [ x,  y, -z], [-x,  y, -z],
        # Bottom (Y-)
        [-x, -y, -z], [ x, -y, -z], [ x, -y,  z], [-x, -y,  z],
        # Right (X+)
        [ x, -y,  z], [ x, -y, -z], [ x,  y, -z], [ x,  y,  z],
        # Left (X-)
        [-x, -y, -z], [-x, -y,  z], [-x,  y,  z], [-x,  y, -z],
    ])

    # 面インデックス (0,1,2, 0,2,3)
    faces = []
    for i in range(6):
        base = i * 4
        faces.append([base, base+1, base+2])
        faces.append([base, base+2, base+3])
    faces = np.array(faces)

    # 4. GLB生成
    material = trimesh.visual.texture.SimpleMaterial(image=atlas)
    visual = trimesh.visual.TextureVisuals(uv=uvs, image=atlas, material=material)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
    
    mesh.export(output_path, file_type='glb')
    print(f"完了: {output_path} を保存しました。")


# ==========================================
# 設定エリア：ここを書き換えてください
# ==========================================
if __name__ == "__main__":
    # 画像ファイルパス
    img_front = "fixed_front.jpg"
    img_back  = "fixed_back.jpg"
    
    # 面の割り当て設定
    # rect: [x開始位置, y開始位置, 幅, 高さ] (すべて0.0〜1.0の割合で指定)
    # source: 0 = img_front, 1 = img_back
    
    mapping_config = {
        # 正面: 画像0の全体を使う
        'front':  {'source': 0, 'rect': [0.0, 0.0, 1.0, 1.0]},
        
        # 背面: 画像1の全体を使う
        'back':   {'source': 1, 'rect': [0.0, 0.0, 1.0, 1.0]},
        
        # 上面: 画像0(表)の上端10%を使う
        'top':    {'source': 0, 'rect': [0.0, 0.0, 1.0, 0.1]},
        
        # 下面: 画像0(表)の下端10%を使う
        'bottom': {'source': 0, 'rect': [0.0, 0.9, 1.0, 0.1]},
        
        # 右側面: 画像0(表)の右端10%を使う
        'right':  {'source': 0, 'rect': [0.9, 0.0, 0.1, 1.0]},
        
        # 左側面: 画像0(表)の左端10%を使う
        'left':   {'source': 0, 'rect': [0.0, 0.0, 0.1, 1.0]},
    }
    
    create_custom_mapped_box(
        img_front, 
        img_back, 
        mapping_config, 
        output_path="mapped_box.glb", 
        thickness=0.2
    )