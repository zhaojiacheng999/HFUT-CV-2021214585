import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_sobel_operator(image):
    # Sobel算子
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # 对图像应用Sobel算子
    sobel_x_img = cv2.filter2D(image, -1, sobel_x)
    sobel_y_img = cv2.filter2D(image, -1, sobel_y)

    # 将Sobel滤波器的结果转换为浮点类型
    sobel_x_img = sobel_x_img.astype(np.float64)
    sobel_y_img = sobel_y_img.astype(np.float64)

    # 计算总的梯度近似值
    sobel_combined = cv2.magnitude(sobel_x_img, sobel_y_img)

    return sobel_combined

# 读取图像
image = cv2.imread('assigment1_test.png', cv2.IMREAD_GRAYSCALE)  # 请替换为您的图像路径
sobel_filtered_image = apply_sobel_operator(image)

# 显示结果
plt.imshow(sobel_filtered_image, cmap='gray')
plt.title('Sobel Filtered Image')
plt.show()

"""
第二步：使用给定卷积核滤波图像
我们将继续使用之前的代码框架，并添加一个函数来应用这个特定的卷积核。
"""

def apply_custom_filter(image):
    # 给定卷积核
    custom_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

    # 对图像应用给定卷积核
    custom_filtered_image = cv2.filter2D(image, -1, custom_kernel)

    return custom_filtered_image

image = cv2.imread('assigment1_test.png', cv2.IMREAD_GRAYSCALE)  # 请替换为您的图像路径
# 应用给定卷积核
custom_filtered_image = apply_custom_filter(image)

# 显示结果
plt.imshow(custom_filtered_image,'gray')
plt.title('Custom Kernel Filtered Image')
plt.show()

"""
第三步：提取并可视化图像的颜色直方图
由于直方图计算不能调用函数包，我们需要手动计算。这里假设图像为灰度图像，如果是彩色图像，您需要对每个颜色通道分别计算直方图。
"""
def plot_histogram(image, color, channel_name):
    # 计算每个通道的直方图
    histogram = np.zeros(256)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            histogram[image[i, j]] += 1

    # 绘制直方图
    plt.plot(histogram, color=color)
    plt.title(f'{channel_name} Channel Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')


# 读取彩色图像
image = cv2.imread('assigment1_test.png')  # 替换为您的图片路径
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 分别计算R, G, B通道的直方图
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plot_histogram(image_rgb[:, :, 0], 'r', 'Red')

plt.subplot(1, 3, 2)
plot_histogram(image_rgb[:, :, 1], 'g', 'Green')

plt.subplot(1, 3, 3)
plot_histogram(image_rgb[:, :, 2], 'b', 'Blue')

plt.tight_layout()
plt.show()

"""
第四步：提取并保存图像的纹理特征
纹理特征的提取方法有很多种，这里我们将使用一个简单的方法：灰度共生矩阵（Gray-Level Co-occurrence Matrix, GLCM）。
"""
def compute_glcm(image, distances, angles):
    # 初始化GLCM矩阵
    glcm = np.zeros((256, 256, len(distances), len(angles)), dtype=np.float64)

    # 计算GLCM
    for d, distance in enumerate(distances):
        for a, angle in enumerate(angles):
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    row = image[i, j]
                    i_offset = int(i + distance * np.sin(angle))
                    j_offset = int(j + distance * np.cos(angle))

                    if i_offset < image.shape[0] and j_offset < image.shape[1]:
                        col = image[i_offset, j_offset]
                        glcm[row, col, d, a] += 1

    # 归一化GLCM
    for d in range(len(distances)):
        for a in range(len(angles)):
            glcm[:, :, d, a] /= glcm[:, :, d, a].sum()

    return glcm

def extract_texture_features(glcm):
    contrast = np.zeros(glcm.shape[2:])
    homogeneity = np.zeros(glcm.shape[2:])
    energy = np.zeros(glcm.shape[2:])
    entropy = np.zeros(glcm.shape[2:])

    for d in range(glcm.shape[2]):
        for a in range(glcm.shape[3]):
            for i in range(glcm.shape[0]):
                for j in range(glcm.shape[1]):
                    contrast[d, a] += glcm[i, j, d, a] * (i - j) ** 2
                    homogeneity[d, a] += glcm[i, j, d, a] / (1 + abs(i - j))
                    energy[d, a] += glcm[i, j, d, a] ** 2
                    if glcm[i, j, d, a] > 0:
                        entropy[d, a] -= glcm[i, j, d, a] * np.log2(glcm[i, j, d, a])

    texture_features = {
        'contrast': contrast,
        'homogeneity': homogeneity,
        'energy': energy,
        'entropy': entropy
    }

    return texture_features

# 计算并提取纹理特征
distances = [1]  # 可以根据需要调整
angles = [0, np.pi/2, np.pi, 3*np.pi/2]  # 四个方向

glcm = compute_glcm(image, distances, angles)
texture_features = extract_texture_features(glcm)

# 保存纹理特征至npy文件
np.save('texture_features.npy', texture_features)

# 打印纹理特征
for feature_name, feature_value in texture_features.items():
    print(f"{feature_name}: \n{feature_value}")




# 加载纹理特征文件
texture_features = np.load('texture_features.npy', allow_pickle=True).item()
def plot_feature_heatmap(feature, title):
    plt.figure(figsize=(6, 6))
    plt.imshow(feature, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.show()

for feature_name, feature_value in texture_features.items():
    plot_feature_heatmap(feature_value, feature_name)

"""
表示的是在不同方向和距离上的对比度值。在这个热图中，每个单元格的颜色对应于对比度的数值大小，其中颜色较深（靠近红色端）的区域表示较高的对比度值，颜色较浅（靠近黄色端）的区域表示较低的对比度值。
热图的轴通常表示不同的参数，
"""