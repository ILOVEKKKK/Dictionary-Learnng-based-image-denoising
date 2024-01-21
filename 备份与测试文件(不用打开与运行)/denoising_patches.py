import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, util, restoration
from skimage.metrics import peak_signal_noise_ratio
from sklearn.feature_extraction import image
from sklearn.decomposition import MiniBatchDictionaryLearning
#批量加载数据集
def load_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg"):
            img_path = os.path.join(folder_path, filename)
            img = io.imread(img_path)
            img = util.img_as_float(img)
            images.append(img)
    return images

# 文件夹中包含18张图片的路径
folder_path = "/dataset"
image_list = load_images(folder_path)

# 合并所有图像为一个数据集
all_data = np.concatenate(image_list, axis=0)

# 从合并的数据集中提取特征
patch_size = (3, 3)
patches = image.extract_patches_2d(all_data, patch_size)
data = patches.reshape(patches.shape[0], -1)

# 进行字典学习
n_components = 100  # Number of dictionary atoms to learn
dl = MiniBatchDictionaryLearning(n_components=n_components, alpha=0.05, n_iter=200)
dl.fit(data)

# 对每张图像进行降噪并计算 PSNR
step=1
for noisy_image in image_list:
    # 提取图像特征小块
    patches = image.extract_patches_2d(noisy_image, patch_size)
    data = patches.reshape(patches.shape[0], -1)

    # 进行图像降噪
    denoised_patches = np.dot(dl.transform(data), dl.components_)
    denoised_patches = denoised_patches.reshape(patches.shape)
    reconstructed_image = image.reconstruct_from_patches_2d(denoised_patches, noisy_image.shape)

    # 计算RGB三通道PSNR
    psnr_values = []

    for channel in range(3):
        noisy_channel = noisy_image[:, :, channel]
        reconstructed_channel = reconstructed_image[:, :, channel]
        psnr_values.append(peak_signal_noise_ratio(noisy_channel, reconstructed_channel))

    # 输出 PSNR
    print(f'PSNR for Image{step}: R_Channel:{psnr_values[0]:.2f},G_Channel:{psnr_values[1]:.2f},B_Channel:{psnr_values[2]:.2f},Average_PSNR{np.mean(psnr_values):.2f}')
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
    plt.text(0.1, 0.5,
             f'PSNR (R): {psnr_values[0]:.2f}\nPSNR (G): {psnr_values[1]:.2f}\nPSNR (B): {psnr_values[2]:.2f}',
             fontsize=10, verticalalignment='center')
    plt.axis('off')
    save_folder_path= "/result_images"
    save_path = os.path.join(save_folder_path, f'reconstructed_image_{step}.png')
    plt.savefig(save_path)  # 保存图像
    # plt.show(block=False)
    step+=1