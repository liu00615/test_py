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
    CREATE TABLE IF NOT EXISTS color (
        id INT AUTO_INCREMENT PRIMARY KEY,
        image_path VARCHAR(255) NOT NULL,
        feature TEXT NOT NULL
    )
""")

# **清空 color 表**
cursor.execute("DELETE FROM color;")  # 删除所有数据
cursor.execute("ALTER TABLE color AUTO_INCREMENT = 1;")  # 重置 ID 计数（可选）
conn.commit()

print("数据库表 color 已清空，准备插入新数据...")


# 提取颜色直方图特征
def extract_color_histogram(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None
    image = cv2.resize(image, (256, 256))  # 统一尺寸
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()  # 归一化并展平
    return hist


# 遍历所有子文件夹
for root, _, files in os.walk(base_folder):
    for filename in files:
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(root, filename)
            feature_vector = extract_color_histogram(image_path)

            if feature_vector is not None:
                feature_json = json.dumps(feature_vector.tolist())  # 转换为 JSON 便于存储
                relative_path = os.path.relpath(image_path, base_folder)  # 存储相对路径

                # 插入数据库
                cursor.execute("INSERT INTO color (image_path, feature) VALUES (%s, %s)", (relative_path, feature_json))

# 提交更改并关闭连接
conn.commit()
cursor.close()
conn.close()

print("所有图像的颜色特征已提取并存入数据库！")