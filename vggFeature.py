import json
import os
import cv2
import mysql.connector
import numpy as np
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

# 目标数据集文件夹
base_folder = "data/256_ObjectCategories"

# 连接数据库
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

# 确保数据库表存在
cursor.execute("""
    CREATE TABLE IF NOT EXISTS vgg (
        id INT AUTO_INCREMENT PRIMARY KEY,
        image_path VARCHAR(255) NOT NULL,
        vgg_features LONGBLOB NOT NULL  # 使用 LONGBLOB 存储序列化的特征
    )
""")

# **清空 vgg 表**
cursor.execute("DELETE FROM vgg;")
cursor.execute("ALTER TABLE vgg AUTO_INCREMENT = 1;")
conn.commit()

print("数据库表 vgg 已清空，准备插入新数据...")

# 手动构建 VGG16 模型
def build_vgg16(input_shape=(224, 224, 3), num_classes=1000):
    # 创建输入层
    inputs = layers.Input(shape=input_shape)

    # 第1组卷积层 (64 filters)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # 第2组卷积层 (128 filters)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # 第3组卷积层 (256 filters)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # 第4组卷积层 (512 filters)
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # 第5组卷积层 (512 filters)
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # 展平层
    x = layers.Flatten()(x)

    # 全连接层
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # 输出层
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # 只返回卷积层输出的模型
    feature_model = Model(inputs=inputs, outputs=x)
    return feature_model

# 加载模型
vgg16_model = build_vgg16()

# 提取图像的特征
def extract_vgg16_features(image):
    # 调整图像尺寸为 224x224 并进行预处理
    img_resized = cv2.resize(image, (224, 224))  # 调整图像大小
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # 转换为 RGB

    # 将图像数组扩展为 (1, 224, 224, 3) 以符合模型输入
    img_array = np.expand_dims(img_rgb, axis=0)  # 增加批次维度
    img_array = img_array / 255.0  # 归一化

    # 提取特征
    features = vgg16_model.predict(img_array)
    return features.flatten()  # 展平特征

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

            # 提取 VGG16 特征
            vgg16_features = extract_vgg16_features(image)

            # 使用 pickle 序列化特征
            serialized_features = pickle.dumps(vgg16_features)

            # 存储图像的路径和 VGG16 特征
            relative_path = os.path.relpath(image_path, base_folder)
            cursor.execute("INSERT INTO vgg (image_path, vgg_features) VALUES (%s, %s)",
                           (relative_path, serialized_features))

# 提交更改并关闭连接
conn.commit()
cursor.close()
conn.close()

print("所有图像的 VGG16 特征已提取并存入数据库！")
