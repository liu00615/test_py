import os
import pickle

import mysql.connector
import numpy as np
from PIL import Image
from scipy.fftpack import dct  # 导入DCT变换函数

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

# 新建数据表
cursor.execute("""
    CREATE TABLE IF NOT EXISTS hash (
        id INT AUTO_INCREMENT PRIMARY KEY,
        image_path VARCHAR(255) NOT NULL,
        aHash BLOB NOT NULL,
        dHash BLOB NOT NULL,
        pHash BLOB NOT NULL
    )
""")

# 清空hash表
cursor.execute("DELETE FROM hash;")
cursor.execute("ALTER TABLE hash AUTO_INCREMENT = 1;")
conn.commit()

print("数据库表 hash 已清空，准备插入新数据...")

# 计算平均哈希 (aHash)
def average_hash(image_path, hash_size=8):
    img = Image.open(image_path).convert('L')  # 转为灰度图
    img = img.resize((hash_size, hash_size), Image.Resampling.LANCZOS)  # 缩放为固定大小
    pixels = np.array(img)
    avg = pixels.mean()  # 计算像素的平均值
    diff = pixels > avg  # 与平均值比较，生成二进制矩阵
    return diff.flatten()

# 计算差异哈希 (dHash)
def difference_hash(image_path, hash_size=9):
    img = Image.open(image_path).convert('L')
    img = img.resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(img)
    diff = pixels[:, 1:] > pixels[:, :-1]  # 比较相邻像素
    return diff.flatten()

# 计算感知哈希 (pHash)
def perceptual_hash(image_path, hash_size=32):
    img = Image.open(image_path).convert('L')
    img = img.resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(img)
    # 计算离散余弦变换 (DCT)
    dct_result = dct(pixels, axis=0)  # 沿列进行 DCT 变换
    dct_result = dct(dct_result, axis=1)  # 沿行进行 DCT 变换
    dct_low_freq = dct_result[:8, :8]  # 选取低频部分
    avg = dct_low_freq.mean()  # 计算低频部分的平均值
    diff = dct_low_freq > avg  # 与平均值比较，生成二进制矩阵
    return diff.flatten()

# 提取图像的哈希特征
def extract_hash_features(image_path):
    a_hash = average_hash(image_path)
    d_hash = difference_hash(image_path)
    p_hash = perceptual_hash(image_path)

    # 使用pickle序列化特征
    a_hash_serialized = pickle.dumps(a_hash)
    d_hash_serialized = pickle.dumps(d_hash)
    p_hash_serialized = pickle.dumps(p_hash)

    return a_hash_serialized, d_hash_serialized, p_hash_serialized

# 遍历所有图像，提取哈希特征并存入数据库
for root, _, files in os.walk(base_folder):
    for filename in files:
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(root, filename)

            # 提取图像哈希特征
            a_hash, d_hash, p_hash = extract_hash_features(image_path)

            # 获取相对路径
            relative_path = os.path.relpath(image_path, base_folder)

            # 将图像路径和哈希特征插入数据库
            cursor.execute("INSERT INTO hash (image_path, aHash, dHash, pHash) VALUES (%s, %s, %s, %s)",
                           (relative_path, a_hash, d_hash, p_hash))

# 提交更改并关闭连接
conn.commit()
cursor.close()
conn.close()

print("所有图像的哈希特征已提取并存入数据库！")
