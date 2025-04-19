import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import random  # 用于随机选择图片

# 定义处理图片的文件夹路径
folder_path = "./d1"  # 本地的 d1 文件夹路径

# 获取文件夹中的所有图片文件
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
small_test_set = random.sample(image_files, 5)  # 随机选择 5 张图片

# 定义 K-Means 聚类的参数
num_clusters = 5  # 聚类的簇数
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)  # 终止条件
attempts = 10  # 尝试运行 K-Means 的次数
flags = cv2.KMEANS_RANDOM_CENTERS  # 随机初始化聚类中心

def kmeans_clustering(image_path, num_clusters):
    """
    对单张图像进行 K 均值聚类
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取文件: {image_path}")
        return None

    # 转换为 RGB 格式（用于 Matplotlib 可视化）
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 将图像数据转换为二维数组
    pixel_values = image.reshape((-1, 3))  # 将图像展平，每个像素有 3 个通道 (R, G, B)
    pixel_values = np.float32(pixel_values)  # 转换为 float32 类型

    # 执行 K-Means 聚类
    _, labels, centers = cv2.kmeans(pixel_values, num_clusters, None, criteria, attempts, flags)

    # 将中心点 (centers) 转换为整数
    centers = np.uint8(centers)

    # 用聚类中心值替换每个像素值
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    # 展示原始图像与聚类结果
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("原始图像")
    plt.imshow(image)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"K-Means 聚类结果 (K = {num_clusters})")
    plt.imshow(segmented_image)
    plt.axis("off")
    plt.show()

# 遍历随机选择的图片进行聚类
for image_file in small_test_set:
    image_path = os.path.join(folder_path, image_file)
    print(f"处理图像: {image_file}")
    kmeans_clustering(image_path, num_clusters)