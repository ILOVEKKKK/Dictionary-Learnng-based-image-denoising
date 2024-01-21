import os
import numpy as np
from sklearnex import patch_sklearn
patch_sklearn() #启动加速补丁

import matplotlib.pyplot as plt
from skimage import io, color, util, restoration
from skimage.metrics import peak_signal_noise_ratio
from sklearn.feature_extraction import image
from sklearn.decomposition import MiniBatchDictionaryLearning

os.environ['CUDA_VISIBLE_DEVICES']='1'

def train_model(image_path, patch_size=(2, 2), n_components=256, alpha=0.3 , n_iter=200):
    # 加载噪声图像
    noisy_image = io.imread(image_path)
    noisy_image = util.img_as_float(noisy_image)

    # 从灰度图中提取特征
    patches = image.extract_patches_2d(noisy_image, patch_size)
    data = patches.reshape(patches.shape[0], -1)

    # 进行字典学习
    dl = MiniBatchDictionaryLearning(n_components=n_components, alpha=alpha, n_iter=n_iter)
    dl.fit(data)

    return dl

def denoise_image(model, noisy_image, patch_size):
    # 提取小块（patches）
    patches = image.extract_patches_2d(noisy_image, patch_size)
    data = patches.reshape(patches.shape[0], -1)

    # 降噪
    denoised_patches = np.dot(model.transform(data), model.components_)
    denoised_patches = denoised_patches.reshape(patches.shape)
    reconstructed_image = image.reconstruct_from_patches_2d(denoised_patches, noisy_image.shape)
    return reconstructed_image

# 指定保存图像的文件夹路径
save_folder_path = "/result_images"

# 文件夹中包含多张图片的路径
folder_path = '/dataset'

# 遍历每张图像进行处理
step=1
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        img_path = os.path.join(folder_path, filename)

        # 训练模型
        trained_model = train_model(img_path)

        # 加载噪声图像
        noisy_image = io.imread(img_path)
        noisy_image = util.img_as_float(noisy_image)

        # 降噪并保存图像
        reconstructed_image = denoise_image(trained_model, noisy_image, patch_size=(2, 2))

        # 计算 PSNR
        psnr_values = []
        for channel in range(3):
            noisy_channel = noisy_image[:, :, channel]
            reconstructed_channel = reconstructed_image[:, :, channel]
            psnr_values.append(peak_signal_noise_ratio(noisy_channel, reconstructed_channel))

        # 输出 PSNR
        # print(f'PSNR for Image {filename}: {np.mean(psnr_values):.2f}')
        print(f'PSNR for Image{step}: R_Channel:{psnr_values[0]:.2f},G_Channel:{psnr_values[1]:.2f},B_Channel:{psnr_values[2]:.2f},Average_PSNR{np.mean(psnr_values):.2f}')

        # 保存图像到指定路径
        save_path = os.path.join(save_folder_path, f'reconstructed_{filename}')
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(noisy_image)
        plt.title('Noised Image')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed_image)
        plt.title('Denoised Image')
        plt.subplot(1, 3, 3)
        plt.text(0.1, 0.5,
                 f'PSNR (R): {psnr_values[0]:.2f}\nPSNR (G): {psnr_values[1]:.2f}\nPSNR (B): {psnr_values[2]:.2f}',
                 fontsize=10, verticalalignment='center')
        plt.axis('off')
        plt.savefig(save_path)  # 保存图像到指定路径
        plt.close()  # 关闭图像窗口
        step+=1