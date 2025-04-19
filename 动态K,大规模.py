import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# 定义处理图片的文件夹路径
folder_path = "./d1"  # 本地的 d1 文件夹路径

# 获取文件夹中的所有图片文件
image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
small_test_set = image_files[:5]  # 少量图片（前 5 张）

# 定义 K-Means 聚类的参数
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)  # 终止条件
attempts = 10  # 尝试运行 K-Means 的次数
flags = cv2.KMEANS_RANDOM_CENTERS  # 随机初始化聚类中心

def resize_image(image, scale=0.5):
    """缩小图片分辨率"""
    height, width = image.shape[:2]
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(image, (new_width, new_height))

def find_best_k(pixel_values, max_k=10):
    """
    动态选择最佳 K 值
    """
    sse = []  # 保存 SSE 值
    silhouette_scores = []  # 保存轮廓系数
    K_values = range(2, max_k + 1)  # 尝试的 K 值范围

    for k in K_values:
        _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, attempts, flags)
        labels = labels.flatten()
        sse.append(np.sum((pixel_values - centers[labels])**2))  # 计算 SSE

        # 计算轮廓系数
        if len(np.unique(labels)) > 1:  # 确保至少有两个簇
            score = silhouette_score(pixel_values, labels)
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(-1)  # 只有一个簇时无效

    # 使用肘部法则选择最佳 K（基于 SSE）
    optimal_k_sse = K_values[np.argmin(np.gradient(sse))]

    # 使用轮廓系数选择最佳 K
    optimal_k_silhouette = K_values[np.argmax(silhouette_scores)]

    # 返回两个方法选择的最佳 K 值（可以根据需要选择其中一个）
    return optimal_k_sse, optimal_k_silhouette, sse, silhouette_scores

def kmeans_clustering(image_path, max_k):
    """
    对单张图像进行 K 均值聚类，并动态选择最佳 K 值
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取文件: {image_path}")
        return None

    # 转换为 RGB 格式（用于 Matplotlib 可视化）
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 缩小图片分辨率
    image = resize_image(image, scale=0.5)

    # 将图像数据转换为二维数组
    pixel_values = image.reshape((-1, 3))  # 将图像展平，每个像素有 3 个通道 (R, G, B)
    pixel_values = np.float32(pixel_values)  # 转换为 float32 类型

    # 动态选择最佳 K 值
    optimal_k_sse, optimal_k_silhouette, sse, silhouette_scores = find_best_k(pixel_values, max_k)

    print(f"最佳 K 值（基于 SSE）: {optimal_k_sse}")
    print(f"最佳 K 值（基于轮廓系数）: {optimal_k_silhouette}")

    # 使用基于轮廓系数的最佳 K 值进行最终聚类
    _, labels, centers = cv2.kmeans(pixel_values, optimal_k_silhouette, None, criteria, attempts, flags)

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
    plt.title(f"K-Means 聚类结果 (K = {optimal_k_silhouette})")
    plt.imshow(segmented_image)
    plt.axis("off")
    plt.show()

    # 返回最佳 K 值和聚类结果
    return optimal_k_silhouette, sse, silhouette_scores

# 遍历小部分图片进行聚类
for image_file in small_test_set:
    image_path = os.path.join(folder_path, image_file)
    print(f"\n处理图像: {image_file}")

    # 执行动态 K 均值聚类
    best_k, sse, silhouette_scores = kmeans_clustering(image_path, max_k=10)