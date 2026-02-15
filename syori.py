import cv2
import numpy as np
import matplotlib.pyplot as plt

def edge_detection_first_derivative(image_path):
    # 1. 画像の読み込み
    img = cv2.imread(image_path)
    
    if img is None:
        print("画像が見つかりません。パスを確認してください。")
        return

    # 2. BGRからRGBへ変換（Matplotlib表示用）
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3. グレースケール変換
    # エッジ検出は通常、色情報ではなく「輝度の変化」を見るため、白黒にします。
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 4. 1次微分 (Sobelフィルタ) の適用
    # cv2.CV_64F を使う理由: 微分値がマイナス（白→黒の変化）になる場合を考慮して浮動小数点型にします。
    
    # x方向の微分（縦のエッジを検出）
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    
    # y方向の微分（横のエッジを検出）
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # 5. 勾配強度（エッジの強さ）の計算
    # ヒュポテニュス（直角三角形の斜辺）計算: sqrt(x^2 + y^2)
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # 6. 表示用に正規化（0-255の範囲に変換）
    # 最大値で割って255を掛け、uint8型に変換
    scale_factor = np.max(gradient_magnitude)
    if scale_factor == 0: scale_factor = 1 # ゼロ除算回避
    
    edge_img = (gradient_magnitude / scale_factor * 255).astype(np.uint8)

    # --- 結果の表示 ---
    plt.figure(figsize=(12, 6))

    # オリジナル画像
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title('Original Image')
    plt.axis('off')

    # エッジ画像
    plt.subplot(1, 2, 2)
    plt.imshow(edge_img, cmap='gray')
    plt.title('Edge Detection (Sobel / 1st Derivative)')
    plt.axis('off')

    plt.show()

    # 画像の保存（必要であれば）
    cv2.imwrite('edge_output.jpg', edge_img)
    print("エッジ画像を保存しました: edge_output.jpg")

# 実行例（'input.jpg'の部分をお手持ちの画像ファイル名に変更してください）
edge_detection_first_derivative('syoyu.jpg')