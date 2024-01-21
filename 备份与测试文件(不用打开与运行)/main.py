import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, util, restoration
from skimage.metrics import peak_signal_noise_ratio
from sklearn.feature_extraction import image
from sklearn.decomposition import MiniBatchDictionaryLearning

# 加载噪声图像
image_path = '../dataset/McM04_noise.jpg'
noisy_image = io.imread(image_path)
noisy_image = util.img_as_float(noisy_image)
print('Noise Image shape:', noisy_image.shape)

# 初始化 PSNR 值数组
psnr_values = np.zeros(3)

#从灰度图中提取特征
patch_size = (2, 2)
patches = image.extract_patches_2d(noisy_image, patch_size)
# print('Number of Patches :', patches.shape[0])
# print('Shape of patches:', patches.shape)
# Reshape the patches for dictionary learning
data = patches.reshape(patches.shape[0], -1)
# print('Shape of Input data :', data.shape)

# 进行字典学习
n_components = 100  # Number of dictionary atoms to learn
dl = MiniBatchDictionaryLearning(n_components=n_components,
                                 alpha=0.05,
                                 n_iter=500)
# training
dl.fit(data)
#
#
# Denoise the patches using the learned dictionary
denoised_patches = np.dot(dl.transform(data), dl.components_)

# Reshape the denoised patches back to their original shape
denoised_patches = denoised_patches.reshape(patches.shape)

# Reconstruct the denoised image from the patches
reconstructed_image = image.reconstruct_from_patches_2d(denoised_patches, noisy_image.shape)


# 遍历每个通道进行处理
for channel in range(3):
    # 提取当前通道的噪声图像
    noisy_channel = noisy_image[:,:,channel]

    # 提取小块（patches）
    # patch_size = (7, 7)
    patch_size = (2, 2)
    patches = image.extract_patches_2d(noisy_channel, patch_size)

    # Reshape the patches for dictionary learning
    data = patches.reshape(patches.shape[0], -1)

    # # Perform dictionary learning
    n_components = 100  # Number of dictionary atoms to learn
    dl = MiniBatchDictionaryLearning(n_components=n_components, alpha=0.05, n_iter=500)
    dl.fit(data)

    # Denoise the patches using the learned dictionary
    denoised_patches = np.dot(dl.transform(data), dl.components_)

    # Reshape the denoised patches back to their original shape
    denoised_patches = denoised_patches.reshape(patches.shape)

    # Reconstruct the denoised channel from the patches
    reconstructed_channel = image.reconstruct_from_patches_2d(denoised_patches, noisy_channel.shape)

    # 计算当前通道的 PSNR 值
    psnr_values[channel] = peak_signal_noise_ratio(noisy_channel, reconstructed_channel)

    # 输出当前通道的 PSNR 值
    if channel==0:
        channel_label="R"
    elif channel==1:
        channel_label="G"
    elif channel==2:
        channel_label="B"
    print(f'PSNR (Channel {channel_label}): {psnr_values[channel]:.2f}')

# 输出整体 PSNR 值
print(f'Average PSNR: {np.mean(psnr_values):.2f}')


# Show the original noisy image and the denoised image
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(noisy_image)
plt.title('Noised Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(reconstructed_image)
plt.title('Denoised Image')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.text(0.1, 0.5, f'PSNR (R): {psnr_values[0]:.2f}\nPSNR (G): {psnr_values[1]:.2f}\nPSNR (B): {psnr_values[2]:.2f}',
         fontsize=10, verticalalignment='center')
plt.axis('off')

plt.show()