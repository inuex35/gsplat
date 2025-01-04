import os
import numpy as np
from transformers import pipeline
from PIL import Image

# パスを指定
dataset = "/workspace/sample"
input_folder = os.path.join(dataset, "images")
output_folder = os.path.join(dataset, "depth")

# 出力フォルダが存在しない場合は作成
os.makedirs(output_folder, exist_ok=True)

# モデルをロード
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf")

# 入力フォルダ内の画像を処理
for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".jpeg", ".png")):  # 対応する画像形式
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")

        # 画像を開く
        image = Image.open(input_path)
        
        # 深度推定を実行
        result = pipe(image)
        depth = result["depth"]  # 深度マップ (PIL Image)
        depth_array = np.array(depth, dtype=np.float32)  # 数値データに変換 (NumPy)

        # 深度値が100以上の部分を1に設定
        depth_array[depth_array >= 100] = 1

        # 深度値を正規化 (0～65535の範囲にスケール)
        depth_min = np.min(depth_array)
        depth_max = np.max(depth_array)
        normalized_depth = ((depth_array - depth_min) / (depth_max - depth_min) * 65535).astype(np.uint16)

        # PNG形式で保存
        depth_image = Image.fromarray(normalized_depth, mode="I;16")  # 16ビットPNGとして保存
        depth_image.save(output_path)
        print(f"Saved: {output_path}")
