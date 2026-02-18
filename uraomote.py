import numpy as np
import trimesh
from PIL import Image

def create_textured_box(front_img_path, back_img_path, output_path="box_model.glb", thickness=0.2):
    """
    正面と背面の画像から、テクスチャ付きの直方体(GLB)を作成する
    thickness: 箱の厚み（画像の横幅を1.0としたときの比率）
    """
    try:
        # 1. 画像の読み込み
        img_front = Image.open(front_img_path).convert("RGBA")
        img_back = Image.open(back_img_path).convert("RGBA")
        
        # 背面画像を正面画像のサイズに合わせる
        img_back = img_back.resize(img_front.size)

        # 2. テクスチャアトラス（統合画像）の作成
        # レイアウト: [正面][背面][側面用カラー]
        w, h = img_front.size
        
        # 側面の色（正面画像の平均色を取得）
        np_img = np.array(img_front)
        avg_color = np_img.mean(axis=(0, 1)).astype(np.uint8)
        
        # アトラスのサイズ決定 (幅は2枚分 + 側面用スペース)
        atlas_w = w * 2 + 100
        atlas_h = h
        
        texture_atlas = Image.new("RGBA", (atlas_w, atlas_h), (0,0,0,0))
        
        # 画像の貼り付け
        texture_atlas.paste(img_front, (0, 0)) # 左: 正面
        texture_atlas.paste(img_back, (w, 0))  # 中: 背面
        
        # 右: 側面用の塗りつぶし領域
        side_color_img = Image.new("RGBA", (100, h), tuple(avg_color))
        texture_atlas.paste(side_color_img, (w * 2, 0))

        # 3. メッシュ（頂点・面・UV）の定義
        # 画像のアスペクト比から箱のサイズを計算
        aspect_ratio = h / w
        box_w = 1.0
        box_h = aspect_ratio
        box_d = thickness # 厚み
        
        # 頂点座標 (24頂点: 6面 × 4頂点) ※UVを分けるために頂点を共有させない
        # X: -0.5 ~ 0.5, Y: -0.5*h ~ 0.5*h, Z: -0.5*d ~ 0.5*d
        x = box_w / 2
        y = box_h / 2
        z = box_d / 2
        
        # 各面の頂点定義 (Counter-Clockwise order)
        # Front (Z+), Back (Z-), Top (Y+), Bottom (Y-), Right (X+), Left (X-)
        vertices = np.array([
            # Front (Z+)
            [-x, -y,  z], [ x, -y,  z], [ x,  y,  z], [-x,  y,  z],
            # Back (Z-) - 背面から見て正しく見えるように配置
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

        # 面データ (各面2つの三角形)
        # 0,1,2, 0,2,3 のパターンを6面分繰り返す
        faces = []
        for i in range(6):
            base = i * 4
            faces.append([base, base+1, base+2])
            faces.append([base, base+2, base+3])
        faces = np.array(faces)

        # UV座標の計算 (テクスチャ上の位置 0.0 ~ 1.0)
        # アトラス全体の幅に対する比率
        u_front_start = 0.0
        u_front_end = w / atlas_w
        
        u_back_start = w / atlas_w
        u_back_end = (w * 2) / atlas_w
        
        u_side_center = (w * 2 + 50) / atlas_w # 側面エリアの中央
        
        # UV配列 (頂点順に対応)
        uvs = np.array([
            # Front (正面画像を使用)
            [u_front_start, 0], [u_front_end, 0], [u_front_end, 1], [u_front_start, 1],
            # Back (背面画像を使用)
            [u_back_start, 0], [u_back_end, 0], [u_back_end, 1], [u_back_start, 1],
            # Top (側面色)
            [u_side_center, 0], [u_side_center, 0], [u_side_center, 0], [u_side_center, 0],
            # Bottom (側面色)
            [u_side_center, 0], [u_side_center, 0], [u_side_center, 0], [u_side_center, 0],
            # Right (側面色)
            [u_side_center, 0], [u_side_center, 0], [u_side_center, 0], [u_side_center, 0],
            # Left (側面色)
            [u_side_center, 0], [u_side_center, 0], [u_side_center, 0], [u_side_center, 0],
        ])

        # 4. Trimeshオブジェクトの作成と保存
        # マテリアルの設定
        material = trimesh.visual.texture.SimpleMaterial(image=texture_atlas)
        visual = trimesh.visual.TextureVisuals(uv=uvs, image=texture_atlas, material=material)
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
        
        # エクスポート
        mesh.export(output_path, file_type='glb')
        print(f"完了: {output_path} を作成しました。")
        print(f"箱のサイズ比率: 幅=1.0, 高さ={box_h:.2f}, 厚み={box_d}")

    except Exception as e:
        print(f"エラーが発生しました: {e}")

# === 実行設定 ===
if __name__ == "__main__":
    # ここに画像ファイル名を指定してください
    # ※ 先ほどの補正済み画像(fixed_front.jpgなど)を使うと綺麗に仕上がります
    input_front = "fixed_front.jpg"
    input_back = "fixed_back.jpg"
    
    # 厚みの設定（0.1 = 横幅の10%の厚み）
    box_thickness = 0.15 
    
    create_textured_box(input_front, input_back, "box_output.glb", thickness=box_thickness)