import json
import os
import numpy as np
import mysql.connector

from keras.src.applications.vgg16 import VGG16
from keras.api.preprocessing import image
from keras.src.applications.vgg16 import preprocess_input
import pickle

from tensorflow.keras import Model
from tensorflow.keras import layers

# 数据库连接配置
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "graduation",
    "connection_timeout": 300
}

# 获取特征的目标数据集文件夹路径
base_folder = "../data/256_ObjectCategories"

# 连接数据库
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

# 确保数据库表存在
cursor.execute("""
    CREATE TABLE IF NOT EXISTS vgg_keras (
        id INT AUTO_INCREMENT PRIMARY KEY,
        image_path VARCHAR(255) NOT NULL,
        vgg_features BLOB NOT NULL  # 使用 BLOB 类型来存储二进制数据
    )
""")

# **清空 vgg_keras 表**
cursor.execute("DELETE FROM vgg_keras;")
cursor.execute("ALTER TABLE vgg_keras AUTO_INCREMENT = 1;")
conn.commit()

print("数据库表 vgg_keras 已清空，准备插入新数据...")

# 加载预训练的 VGG16 模型
def build_vgg16(input_shape=(224, 224, 3)):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    # 提取卷积层的输出作为特征
    x = layers.GlobalAveragePooling2D()(base_model.output)
    feature_model = Model(inputs=base_model.input, outputs=x)
    return feature_model

# 加载模型
vgg16_model = build_vgg16()

# 提取图像的特征
def extract_vgg16_features(image_path):
    # 加载并预处理图像
    img = image.load_img(image_path, target_size=(224, 224))  # 调整为模型输入大小
    img_array = image.img_to_array(img)  # 转换为数组
    img_array = np.expand_dims(img_array, axis=0)  # 增加批次维度
    img_array = img_array / 255.0  # 归一化

    # 提取特征
    features = vgg16_model.predict(img_array)
    return features.flatten()  # 展平特征

# 遍历所有子文件夹，处理每张图像
for root, _, files in os.walk(base_folder):
    for filename in files:
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(root, filename)

            # 提取 VGG16 特征
            vgg16_features = extract_vgg16_features(image_path)

            # 使用 pickle 库序列化特征
            vgg16_features_serialized = pickle.dumps(vgg16_features)

            # 获取相对路径
            relative_path = os.path.relpath(image_path, base_folder)

            # 将图像路径和 VGG16 特征插入数据库
            cursor.execute("INSERT INTO vgg_keras (image_path, vgg_features) VALUES (%s, %s)",
                           (relative_path, vgg16_features_serialized))

# 提交更改并关闭连接
conn.commit()
cursor.close()
conn.close()

print("所有图像的 VGG16 特征已提取并存入数据库！")
