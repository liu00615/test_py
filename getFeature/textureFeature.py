import json
import os

import cv2
import mysql.connector
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern

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

# 创建表（如果不存在）
cursor.execute("""
    CREATE TABLE IF NOT EXISTS texture (
        id INT AUTO_INCREMENT PRIMARY KEY,
        image_path VARCHAR(255) NOT NULL,
        glcm_features TEXT NOT NULL,
        lbp_features TEXT NOT NULL
    )
""")

# 清空数据表
cursor.execute("DELETE FROM texture;")
cursor.execute("ALTER TABLE texture AUTO_INCREMENT = 1;")
conn.commit()

print("数据库表 texture 已清空，准备插入新数据...")


# 提取 GLCM 特征
def extract_glcm_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray_image, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], symmetric=True, normed=True)

    # 提取 GLCM 特征
    features = [
        graycoprops(glcm, 'contrast').mean(),
        graycoprops(glcm, 'correlation').mean(),
        graycoprops(glcm, 'energy').mean(),
        graycoprops(glcm, 'homogeneity').mean()
    ]

    return features


# 提取 LBP 特征
def extract_lbp_features(image, radius=1, n_points=8):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')

    # 计算 LBP 直方图
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist /= (lbp_hist.sum() + 1e-6)  # 归一化

    return lbp_hist.tolist()


# 遍历所有图像文件
for root, _, files in os.walk(base_folder):
    for filename in files:
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(root, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图像: {image_path}")
                continue

            # 提取特征
            glcm_features = extract_glcm_features(image)
            lbp_features = extract_lbp_features(image)

            # 存入数据库
            cursor.execute("INSERT INTO texture (image_path, glcm_features, lbp_features) VALUES (%s, %s, %s)",
                           (os.path.relpath(image_path, base_folder), json.dumps(glcm_features),
                            json.dumps(lbp_features)))

# 提交更改
conn.commit()
cursor.close()
conn.close()

print("所有图像的 GLCM 和 LBP 特征已提取并存入数据库！")