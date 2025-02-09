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
    "database": "graduation"
}

# 目标数据集文件夹
base_folder = "data/256_ObjectCategories"  # 数据集的根目录

# 连接数据库
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

# 确保数据库表存在
cursor.execute("""
    CREATE TABLE IF NOT EXISTS edge (
        id INT AUTO_INCREMENT PRIMARY KEY,
        image_path VARCHAR(255) NOT NULL,
        hu_moments TEXT NOT NULL,
        hog_features TEXT NOT NULL
    )
""")

# **清空 edge 表**
cursor.execute("DELETE FROM edge;")  # 删除所有数据
cursor.execute("ALTER TABLE edge AUTO_INCREMENT = 1;")  # 重置 ID 计数（可选）
conn.commit()

print("数据库表 edge 已清空，准备插入新数据...")


# 形状不变矩特征提取函数（Hu Moments）
def extract_hu_moments(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    moments = cv2.moments(thresh)
    hu_moments = cv2.HuMoments(moments)
    return hu_moments.flatten()


# 方向梯度直方图特征提取函数（HOG）
def extract_hog_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None
    # HOG特征提取
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(image)
    return hog_features.flatten()


# 遍历所有子文件夹
for root, _, files in os.walk(base_folder):
    for filename in files:
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(root, filename)

            # 提取形状不变矩特征
            hu_moments = extract_hu_moments(image_path)
            if hu_moments is None:
                continue

            # 提取方向梯度直方图特征
            hog_features = extract_hog_features(image_path)
            if hog_features is None:
                continue

            # 将特征转换为 JSON 格式以便存储
            hu_moments_json = json.dumps(hu_moments.tolist())
            hog_features_json = json.dumps(hog_features.tolist())

            # 存储图像的路径、Hu Moments 和 HOG 特征
            relative_path = os.path.relpath(image_path, base_folder)  # 存储相对路径
            cursor.execute("INSERT INTO edge (image_path, hu_moments, hog_features) VALUES (%s, %s, %s)",
                           (relative_path, hu_moments_json, hog_features_json))

# 提交更改并关闭连接
conn.commit()
cursor.close()
conn.close()

print("所有图像的边缘特征已提取并存入数据库！")
