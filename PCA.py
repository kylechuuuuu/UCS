import torch
import numpy as np
import cv2
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pdb
# 假设 mask_tokens_out 是你从模型中得到的特征数据
# mask_tokens_out 的形状为 (batch_size, num_masks, feature_dim)
# 例如，我们使用随机数据作为示例
batch_size = 5
num_masks = 5
feature_dim = 100
mask_tokens_out = torch.rand(batch_size, num_masks, feature_dim)

# 提取第 i 个 mask 的特征数据
i = 0
features = mask_tokens_out[:, i, :].numpy()

# 使用 PCA 提取主成分
pca = PCA(n_components=5)  # 提取前两个主成分以便于可视化
principal_components = pca.fit_transform(features)

# 创建文件夹保存图片
output_folder = 'output_images'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 定义保存图片的函数
def save_image(data, filename, title):
    # Normalize data to [0, 255]
    data_normalized = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
    data_normalized = data_normalized.astype(np.uint8)
    
    # Apply binary threshold
    _, binary_image = cv2.threshold(data_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Save image
    cv2.imwrite(filename, binary_image)
    
    # Optional: Save with matplotlib to include the title
    plt.imshow(binary_image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.savefig(filename.replace('.png', '_with_title.png'), bbox_inches='tight')
    plt.close()

# 保存原始特征数据二值化图像
original_filename = os.path.join(output_folder, 'original_features.png')
save_image(features, original_filename, 'Original Features')

# 保存PCA结果二值化图像
pca_filename = os.path.join(output_folder, 'pca_features.png')
save_image(principal_components, pca_filename, 'PCA Features')

print(f"Images saved in {output_folder} folder.")