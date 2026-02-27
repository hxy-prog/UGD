import numpy as np
from PIL import Image

# 加载图像，并确保它是二值化的
img_path = 'image1.png'  # 替换为你的图片路径
image = Image.open(img_path).convert('L')  # 转换为灰度图像
image_np = np.array(image, dtype=np.float32) / 255.0  # 归一化到[0, 1]
image_np = np.where(image_np > 0.5, 1.0, 0.0)  # 确保图像被二值化

# 定义一个函数来添加噪声并重新二值化
def add_noise(image, noise_level):
    """根据给定的噪声水平向图像添加高斯噪声"""
    noise = np.random.normal(0, noise_level, image.shape)
    noisy_image = image + noise
    # 重新二值化图像
    noisy_image = np.where(noisy_image > 0.5, 1.0, 0.0)
    return noisy_image

# 设置不同的噪声级别
noise_levels = [0.15, 0.5, 0.9]

# 对每个噪声级别应用加噪并保存结果
for i, level in enumerate(noise_levels):
    noisy_image = add_noise(image_np, level)
    noisy_image_pil = Image.fromarray((noisy_image * 255).astype(np.uint8))
    
    # 保存加噪后的图像
    save_path = f'noisy_image_level_{i+1}.png'
    noisy_image_pil.save(save_path)
    print(f"已保存: {save_path}")

# 最后，可以保存原始图像作为对比（可选）
original_save_path = 'original_image.png'
image.save(original_save_path)
print(f"已保存: {original_save_path}")