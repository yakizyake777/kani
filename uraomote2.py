import numpy as np
import trimesh
from PIL import Image

def create_edge_textured_box(front_img_path, back_img_path, output_path="box_model.glb", thickness=0.2):
    """
    正面と背面の画像から直方体を作成し、
    側面のテクスチャを画像の端のピクセルから生成して貼り付ける
    """
    try:
        # 1. 画像の読み込み
        img_front = Image.open(front_img_path).convert("RGBA")
        img_back = Image.open(back_img_path).convert("RGBA")
        
        # 背面画像を正面画像のサイズに合わせる
        img_back = img_back.resize(img_front.size)

        w, h = img_front.size

        # 2. 側面用テクスチャの生成（画像の端1ピクセルを取得して引き伸ばす）
        # Top: 正面上端
        edge_top = img_front.crop((0, 0, w, 1))
        # Bottom: 正面下端
        edge_bottom = img_front.crop((0, h-1, w, h))
        # Left: 正面左端
        edge_left = img_front.crop((0, 0, 1, h))
        # Right: 正面右端
        edge_right = img_front.crop((w-1, 0, w, h))

        # 3. テクスチャアトラスの作成
        # レイアウト: [Front][Back][Top][Bottom][Right][Left]
        # 側面画像はUVマッピングしやすいように配置する
        
        # アトラスの幅計算 (Front + Back + 側面用予備スペース)
        # 側面テクスチャは細長いので、アトラスの下に並べるか横に並べるか。
        # ここではシンプルに横に並べます。
        # ただし側面(Left/Right)は縦長なので、90度回転させて横長にして配置します。
        
        atlas_w = w * 2 + max(w, h) * 4 # 十分な幅を確保
        atlas_h = max(w, h)
        
        texture_atlas = Image.new("RGBA", (atlas_w, atlas_h), (0,0,0,0))
        
        # 配置位置の定義 (x座標)
        pos_front = 0
        pos_back = w
        pos_top = w * 2
        pos_bottom = pos_top + w
        pos_right = pos_bottom + w
        pos_left = pos_right + h # Right/Leftは回転させるので幅がhになる

        # 貼り付け
        texture_atlas.paste(img_front, (pos_front, 0))
        texture_atlas.paste(img_back, (pos_back, 0))
        
        # Top/Bottomはそのまま横長として配置
        texture_atlas.paste(edge_top.resize((w, int(atlas_h*0.1))), (pos_top, 0)) # 少し太くして視認性確保
        texture_atlas.paste(edge_bottom.resize((w, int(atlas_h*0.1))), (pos_bottom, 0))
        
        # Right/Leftは縦長なので、90度回転させて横長にして配置 (UV側で調整する手もあるが画像側で回すのが楽)
        # Right: 上がTop側。90度回転させると左がTop側になる。
        img_right_rotated = edge_right.rotate(90, expand=True)
        texture_atlas.paste(img_right_rotated.resize((h, int(atlas_h*0.1))), (pos_right, 0))
        
        # Left: 上がTop側。90度回転
        img_left_rotated = edge_left.rotate(90, expand=True)
        texture_atlas.paste(img_left_rotated.resize((h, int(atlas_h*0.1))), (pos_left, 0))

        # 4. メッシュ定義
        aspect_ratio = h / w
        box_w = 1.0
        box_h = aspect_ratio
        box_d = thickness
        
        x, y, z = box_w/2, box_h/2, box_d/2
        
        vertices = np.array([
            # Front (Z+)
            [-x, -y,  z], [ x, -y,  z], [ x,  y,  z], [-x,  y,  z],
            # Back (Z-)
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

        faces = []
        for i in range(6):
            base = i * 4
            faces.append([base, base+1, base+2])
            faces.append([base, base+2, base+3])
        faces = np.array(faces)

        # 5. UV座標の計算
        def get_uv_rect(x_start, width, total_w, y_start=0, height_ratio=1.0):
            u0 = x_start / total_w
            u1 = (x_start + width) / total_w
            v0 = 0.0
            v1 = height_ratio # アトラスの高さに対する使用率
            return u0, u1, v0, v1

        # 各エリアのUV範囲を取得
        # 側面画像は atlas_h * 0.1 の高さに引き伸ばして貼り付けたので、V座標もそれに合わせる
        side_v_ratio = 0.1 

        uf0, uf1, vf0, vf1 = get_uv_rect(pos_front, w, atlas_w, 0, 1.0)
        ub0, ub1, vb0, vb1 = get_uv_rect(pos_back, w, atlas_w, 0, 1.0)
        
        ut0, ut1, vt0, vt1 = get_uv_rect(pos_top, w, atlas_w, 0, side_v_ratio)
        ubo0, ubo1, vbo0, vbo1 = get_uv_rect(pos_bottom, w, atlas_w, 0, side_v_ratio)
        
        ur0, ur1, vr0, vr1 = get_uv_rect(pos_right, h, atlas_w, 0, side_v_ratio)
        ul0, ul1, vl0, vl1 = get_uv_rect(pos_left, h, atlas_w, 0, side_v_ratio)

        uvs = np.array([
            # Front
            [uf0, 1], [uf1, 1], [uf1, 0], [uf0, 0],
            # Back
            [ub0, 1], [ub1, 1], [ub1, 0], [ub0, 0],
            
            # Top (画像の上端ピクセル列)
            # 頂点順: 左手前(-x,y,z), 右手前(x,y,z), 右奥(x,y,-z), 左奥(-x,y,-z)
            # 画像は左から右へ並んでいるので、そのままマッピング
            [ut0, vt1], [ut1, vt1], [ut1, vt0], [ut0, vt0],
            
            # Bottom (画像の下端ピクセル列)
            # 頂点順: 左手前, 右手前, 右奥, 左奥 (Yマイナス面なので底から見て)
            [ubo0, vbo1], [ubo1, vbo1], [ubo1, vbo0], [ubo0, vbo0],
            
            # Right (画像の右端ピクセル列)
            # 画像を90度回転させて横長にした。
            # 元画像の「上」がアトラスの「左」、「下」が「右」。
            # 頂点順: 手前上, 手前下, 奥下, 奥上 (Xプラス面)
            # マッピング: 左(元の上) -> 右(元の下)
            [ur0, vr1], [ur1, vr1], [ur1, vr0], [ur0, vr0],
            
            # Left (画像の左端ピクセル列)
            # 頂点順: 手前下, 手前上, 奥上, 奥下 (Xマイナス面)
            # 画像は90度回転(左が上、右が下)
            # ここは回転方向と頂点順序の合わせが複雑だが、一旦標準マッピング
            [ul1, vl1], [ul0, vl1], [ul0, vl0], [ul1, vl0],
        ])

        # 6. 保存
        material = trimesh.visual.texture.SimpleMaterial(image=texture_atlas)
        visual = trimesh.visual.TextureVisuals(uv=uvs, image=texture_atlas, material=material)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, visual=visual, process=False)
        
        mesh.export(output_path, file_type='glb')
        print(f"完了: {output_path}")

    except Exception as e:
        print(f"エラー: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 画像ファイル名を指定
    input_front = "fixed_front.jpg"
    input_back = "fixed_back.jpg"
    
    # 実行
    create_edge_textured_box(input_front, input_back, "edge_textured_box.glb", thickness=0.15)