import os
import cv2
import numpy as np
import mysql.connector
import json

# 配置数据库连接
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "graduation",
    "connection_timeout": 300  # 解决数据库超时问题
}

# 目标数据集文件夹
base_folder = "data/256_ObjectCategories"

# 连接数据库
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

# 确保数据库表存在（使用 MEDIUMTEXT 存储 HOG 特征）
cursor.execute("""
    CREATE TABLE IF NOT EXISTS edge (
        id INT AUTO_INCREMENT PRIMARY KEY,
        image_path VARCHAR(255) NOT NULL,
        hu_moments TEXT NOT NULL,
        hog_features MEDIUMTEXT NOT NULL  # 允许更大的存储空间
    )
""")

# **清空 edge 表**
cursor.execute("DELETE FROM edge;")
cursor.execute("ALTER TABLE edge AUTO_INCREMENT = 1;")
conn.commit()

print("数据库表 edge 已清空，准备插入新数据...")


# 形状不变矩特征提取（Hu Moments）
def extract_hu_moments(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(thresh)
    hu_moments = cv2.HuMoments(moments)
    return hu_moments.flatten()


# 方向梯度直方图（HOG）特征提取（减少特征维度）
def extract_hog_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None
    image = cv2.resize(image, (128, 128))  # 降低计算复杂度
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(image)

    if hog_features is not None:
        hog_features = hog_features.flatten()

        # 降维（只取前 500 维）
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

            # 提取 HOG 特征
            hog_features = extract_hog_features(image_path)
            if hog_features is None:
                continue

            # 转换为 JSON 存储
            hu_moments_json = json.dumps(hu_moments.tolist())
            hog_features_json = json.dumps(hog_features.tolist())

            # 存储图像的路径、Hu Moments 和 HOG 特征
            relative_path = os.path.relpath(image_path, base_folder)
            cursor.execute("INSERT INTO edge (image_path, hu_moments, hog_features) VALUES (%s, %s, %s)",
                           (relative_path, hu_moments_json, hog_features_json))

# 提交更改并关闭连接
conn.commit()
cursor.close()
conn.close()

print("所有图像的边缘特征已提取并存入数据库！")
