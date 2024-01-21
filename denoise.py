import os
import numpy as np
# from sklearnex import patch_sklearn
# patch_sklearn() #启动加速补丁,加速字典学习速度

import matplotlib.pyplot as plt
from skimage import io, color, util, restoration
from skimage.metrics import peak_signal_noise_ratio
from sklearn.feature_extraction import image
from sklearn.decomposition import MiniBatchDictionaryLearning

os.environ['CUDA_VISIBLE_DEVICES']='1'

def train_model(image_path, patch_size=(2, 2), n_components=256, alpha=0.05 , n_iter=500):
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
    # 提取图像特征小块
    patches = image.extract_patches_2d(noisy_image, patch_size)
    data = patches.reshape(patches.shape[0], -1)

    # 进行图像降噪
    denoised_patches = np.dot(model.transform(data), model.components_)
    denoised_patches = denoised_patches.reshape(patches.shape)
    reconstructed_image = image.reconstruct_from_patches_2d(denoised_patches, noisy_image.shape)
    return reconstructed_image

def channel_training():
    patch_size = (2, 2)
    patches = image.extract_patches_2d(noisy_channel, patch_size)
    data = patches.reshape(patches.shape[0], -1)
    n_components = 256  # Number of dictionary atoms to learn
    dl = MiniBatchDictionaryLearning(n_components=n_components, alpha=0.05, n_iter=500)
    dl.fit(data)
    denoised_patches = np.dot(dl.transform(data), dl.components_)

    # 变换噪声图像尺寸
    denoised_patches = denoised_patches.reshape(patches.shape)

    # 从噪声图像重构通道
    reconstructed_channel = image.reconstruct_from_patches_2d(denoised_patches, noisy_channel.shape)
    return reconstructed_channel,dl

# 指定保存图像的文件夹路径
save_folder_path = "D:/task4_denoising/cvx_opt/test"

# 文件夹中包含多张图片的路径
folder_path = 'D:/task4_denoising/cvx_opt/dataset'

# 遍历每张图像进行处理
step=1
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg"):
        img_path = os.path.join(folder_path, filename)

        # 从噪声图像中训练字典学习模型
        trained_model = train_model(img_path)

        # 加载噪声图像
        noisy_image = io.imread(img_path)
        noisy_image = util.img_as_float(noisy_image)

        # 降噪并保存图像
        reconstructed_image = denoise_image(trained_model, noisy_image, patch_size=(2, 2))

        # 计算 各个通道PSNR
        psnr_values = []
        for channel in range(3):
            noisy_channel = noisy_image[:, :, channel]
            reconstructed_channel,model=channel_training()
            psnr_values.append(peak_signal_noise_ratio(noisy_channel, reconstructed_channel))

        # 输出 各个通道PSNR
        print(f'PSNR for Image{step}: R_Channel:{psnr_values[0]:.2f},G_Channel:{psnr_values[1]:.2f},B_Channel:{psnr_values[2]:.2f},Average_PSNR{np.mean(psnr_values):.2f}')


        # 保存图像到指定路径
        save_path = os.path.join(save_folder_path, f'reconstructed_{filename}')
        plt.figure(figsize=(12, 5))#创建宽度12，高度5的画布来展示降噪效果
        plt.subplot(1, 3, 1)#子图1显示原始含噪声图像
        plt.imshow(noisy_image)
        plt.title('Noised Image',fontsize=15)
        plt.axis('off')
        plt.subplot(1, 3, 2)#子图2显示降噪后图像
        plt.imshow(reconstructed_image)
        plt.title('Denoised Image',fontsize=15)
        plt.axis('off')
        plt.subplot(1, 3, 3)#子图3显示降噪后三个通道的psnr值
        plt.text(0.1, 0.5,
                 f'Channel(R) PSNR: {psnr_values[0]:.2f}\nChannel(G) PSNR: {psnr_values[1]:.2f}\nChannel(B) PSNR: {psnr_values[2]:.2f}',
                 fontsize=20, verticalalignment='center')
        plt.axis('off')
        plt.savefig(save_path)  # 保存图像到指定路径
        plt.close()  # 关闭图像窗口
        step+=1