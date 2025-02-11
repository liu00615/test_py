import os
import cv2
import numpy as np
import mysql.connector
import json
from skimage.feature import local_binary_pattern
from skimage.feature import graycomatrix, graycoprops  # 新版不用grey***
from skimage import data

# 数据库连接配置
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "graduation",
    "connection_timeout": 300
}

# 目标数据集文件夹
base_folder = "../data/256_ObjectCategories"

# 连接数据库
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

# 确保数据库表存在
cursor.execute("""
    CREATE TABLE IF NOT EXISTS texture (
        id INT AUTO_INCREMENT PRIMARY KEY,
        image_path VARCHAR(255) NOT NULL,
        glcm_features TEXT NOT NULL,  # 存储灰度共生矩阵特征
        lbp_features TEXT NOT NULL  # 存储LBP特征
    )
""")

# **清空 texture 表**
cursor.execute("DELETE FROM texture;")
cursor.execute("ALTER TABLE texture AUTO_INCREMENT = 1;")
conn.commit()

print("数据库表 texture 已清空，准备插入新数据...")

# 提取灰度共生矩阵特征（GLCM）
def extract_glcm_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 计算灰度共生矩阵，考虑距离为1，方向为0度（水平）、45度、90度、135度
    glcm = graycomatrix(gray_image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], symmetric=True, normed=True)

    # 提取几个常见的灰度共生矩阵特征
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    return [contrast, correlation, energy, homogeneity]


# 提取局部二值模式（LBP）特征
def extract_lbp_features(image, radius=1, n_points=8):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')

    # 计算LBP的直方图
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist /= (lbp_hist.sum() + 1e-6)  # 归一化处理

    return lbp_hist


# 遍历所有子文件夹
for root, _, files in os.walk(base_folder):
    for filename in files:
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(root, filename)

            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图像: {image_path}")
                continue

            # 提取 GLCM 特征
            glcm_features = extract_glcm_features(image)

            # 提取 LBP 特征
            lbp_features = extract_lbp_features(image)

            # 转换为 JSON 格式存储
            glcm_features_json = json.dumps(glcm_features)
            lbp_features_json = json.dumps(lbp_features.tolist())  # 将 LBPhist 转换为列表存储

            # 存储图像的路径、GLCM 特征和 LBP 特征
            relative_path = os.path.relpath(image_path, base_folder)
            cursor.execute("INSERT INTO texture (image_path, glcm_features, lbp_features) VALUES (%s, %s, %s)",
                           (relative_path, glcm_features_json, lbp_features_json))

# 提交更改并关闭连接
conn.commit()
cursor.close()
conn.close()

print("所有图像的纹理特征（GLCM 和 LBP）已提取并存入数据库！")
