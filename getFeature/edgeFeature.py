import json
import os

import cv2
import mysql.connector
import numpy as np
from sklearn.preprocessing import StandardScaler

# 配置数据库连接
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "graduation",
    "connection_timeout": 300  # 解决数据库超时问题
}

# 目标数据集文件夹
base_folder = "../data/256_ObjectCategories"

# 连接数据库
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

# 新建数据表 使用MEDIUMTEXT存储HOG特征
cursor.execute(""" 
    CREATE TABLE IF NOT EXISTS edge (
        id INT AUTO_INCREMENT PRIMARY KEY,
        image_path VARCHAR(255) NOT NULL,
        hu_moments TEXT NOT NULL,
        hog_features MEDIUMTEXT NOT NULL  # 可以存放较长的数据
    )
""")

# 清空edge表
cursor.execute("DELETE FROM edge;")
cursor.execute("ALTER TABLE edge AUTO_INCREMENT = 1;")
conn.commit()

print("数据库表 edge 已清空，准备插入新数据...")


# 图像预处理：去噪、归一化和图像缩放
def preprocess_image(image, target_size=(128, 128)):
    # 高斯滤波去噪
    image = cv2.GaussianBlur(image, (5, 5), 0)
    # 将图像缩放到目标尺寸
    image = cv2.resize(image, target_size)
    return image

def extract_hu_moments(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None

    image = preprocess_image(image)  # 预处理图像

    # 使用Otsu阈值法自动计算最佳阈值
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Otsu 二值化
    moments = cv2.moments(thresh)
    hu_moments = cv2.HuMoments(moments).flatten()

    # 归一化Hu Moments
    hu_moments = np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-6)  # 防止对数为负无穷

    # 标准化Hu Moments
    scaler = StandardScaler()
    hu_moments = scaler.fit_transform(hu_moments.reshape(-1, 1)).flatten()

    return hu_moments


# 方向梯度直方图特征提取（HOG）
def extract_hog_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None

    image = preprocess_image(image)  # 预处理图像
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(image)

    if hog_features is not None:
        hog_features = hog_features.flatten()

        # 降维
        if len(hog_features) > 500:
            hog_features = hog_features[:500]

    return hog_features


# 遍历所有子文件夹
for root, _, files in os.walk(base_folder):
    for filename in files:
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(root, filename)

            # 提取形状不变矩特征
            hu_moments = extract_hu_moments(image_path)
            if hu_moments is None:
                continue

            # 提取HOG特征
            hog_features = extract_hog_features(image_path)
            if hog_features is None:
                continue

            # 转换为JSON存储
            hu_moments_json = json.dumps(hu_moments.tolist())
            hog_features_json = json.dumps(hog_features.tolist())

            # 存储图像的路径、Hu Moments和HOG特征
            relative_path = os.path.relpath(image_path, base_folder)
            cursor.execute("INSERT INTO edge (image_path, hu_moments, hog_features) VALUES (%s, %s, %s)",
                           (relative_path, hu_moments_json, hog_features_json))

# 提交更改并关闭连接
conn.commit()
cursor.close()
conn.close()

print("所有图像的边缘特征已提取并存入数据库！")
