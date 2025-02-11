import os
import cv2
import numpy as np
import mysql.connector
import pickle

# 数据库连接配置
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "graduation",
    "connection_timeout": 300
}

# 目标数据集文件夹
base_folder = "data/256_ObjectCategories"

# 连接数据库
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

# 确保数据库表存在
cursor.execute("""
    CREATE TABLE IF NOT EXISTS sift (
        id INT AUTO_INCREMENT PRIMARY KEY,
        image_path VARCHAR(255) NOT NULL,
        sift_features LONGBLOB NOT NULL  # 存储序列化的SIFT特征(以二进制形式存储)
    )
""")

# **清空 sift 表**
cursor.execute("DELETE FROM sift;")
cursor.execute("ALTER TABLE sift AUTO_INCREMENT = 1;")
conn.commit()

print("数据库表 sift 已清空，准备插入新数据...")

# 初始化 SIFT 算法
sift = cv2.SIFT_create()


# 提取 SIFT 特征的函数
def extract_sift_features(image):
    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测关键点并计算描述符
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)

    return descriptors


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

            # 提取 SIFT 特征
            sift_features = extract_sift_features(image)

            # 如果没有检测到特征点，则跳过
            if sift_features is None:
                print(f"图像 {image_path} 没有检测到特征点，跳过...")
                continue

            # 使用 pickle 将 SIFT 特征序列化
            sift_features_pickle = pickle.dumps(sift_features)

            # 存储图像的路径和序列化的 SIFT 特征
            relative_path = os.path.relpath(image_path, base_folder)
            cursor.execute("""
                INSERT INTO sift (image_path, sift_features)
                VALUES (%s, %s)
            """, (relative_path, sift_features_pickle))

            print(f"图像 {image_path} 特征提取成功！")

# 提交更改并关闭连接
conn.commit()
cursor.close()
conn.close()

print("所有图像的 SIFT 特征已提取并存入数据库！")
