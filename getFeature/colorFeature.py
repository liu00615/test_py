import json
import os

import cv2
import mysql.connector

# 配置数据库连接
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

# 新建数据表 使用MEDIUMTEXT存储颜色直方图特征
cursor.execute("""
    CREATE TABLE IF NOT EXISTS color (
        id INT AUTO_INCREMENT PRIMARY KEY,
        image_path VARCHAR(255) NOT NULL,
        hsv_moments TEXT NOT NULL,
        color_histogram MEDIUMTEXT NOT NULL  
    )
""")

# 清空color表，方便调试
cursor.execute("DELETE FROM color;")
cursor.execute("ALTER TABLE color AUTO_INCREMENT = 1;")
conn.commit()

print("数据库表 color 已清空，准备插入新数据...")

# 预处理步骤：图像缩放、去噪和直方图均衡化
def preprocess_image(image_path, target_size=(256, 256)):
    image = cv2.imread(image_path)
    if image is None:
        print(f"无法读取图像: {image_path}")
        return None

    # 缩放图像到统一尺寸
    image = cv2.resize(image, target_size)

    # 去噪（高斯滤波）
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # 转换为HSV颜色空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 对亮度通道V进行直方图均衡化
    hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])

    return hsv_image

# 提取 HSV 中心矩特征
def extract_hsv_moments(image_path):
    hsv_image = preprocess_image(image_path)
    if hsv_image is None:
        return None

    # 提取亮度通道V
    v_channel = hsv_image[:, :, 2]

    # 计算V通道的矩
    moments = cv2.moments(v_channel)
    hsv_moments = cv2.HuMoments(moments).flatten()
    return hsv_moments


# 提取颜色直方图特征（48维）
def extract_color_histogram(image_path):
    hsv_image = preprocess_image(image_path)
    if hsv_image is None:
        return None

    # 每个通道6bins，共6*6*6=216维
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [6, 6, 6], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()  # 归一化并展平为一维数组

    return hist[:48]  # 48维特征


# 遍历所有子文件夹
for root, _, files in os.walk(base_folder):
    for filename in files:
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(root, filename)

            # 提取HSV中心矩特征
            hsv_moments = extract_hsv_moments(image_path)
            if hsv_moments is None:
                continue

            # 提取颜色直方图特征
            color_histogram = extract_color_histogram(image_path)
            if color_histogram is None:
                continue

            # 转换为JSON存储
            hsv_moments_json = json.dumps(hsv_moments.tolist())
            color_histogram_json = json.dumps(color_histogram.tolist())

            # 存储图像的路径、HSV中心矩特征和颜色直方图特征
            relative_path = os.path.relpath(image_path, base_folder)
            cursor.execute("INSERT INTO color (image_path, hsv_moments, color_histogram) VALUES (%s, %s, %s)",
                           (relative_path, hsv_moments_json, color_histogram_json))

# 提交更改并关闭连接
conn.commit()
cursor.close()
conn.close()

print("所有图像的颜色特征已提取并存入数据库！")
