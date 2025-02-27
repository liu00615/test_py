import pickle

import cv2
import mysql.connector
import numpy as np
from flask import Flask, request, jsonify, Blueprint
from scipy.spatial.distance import cdist
from tensorflow.keras import Model
from tensorflow.keras import layers
# from vggFeature import build_vgg16 # 导入会出发vggFeature自动执行，舍弃

# 定义Flask蓝图
search_by_vgg_my_route = Blueprint('searchbyvggmy', __name__)

# 数据库连接配置
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "graduation",
    "connection_timeout": 300
}

# 连接数据库
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()


# 自定义VGG16模型
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
    # 调整图像尺寸为224x224并进行预处理
    img_resized = cv2.resize(image, (224, 224))  # 调整图像大小
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)  # 转换为RGB

    # 将图像数组扩展为(1, 224, 224, 3)以符合模型输入
    img_array = np.expand_dims(img_rgb, axis=0)  # 增加批次维度
    img_array = img_array / 255.0  # 归一化

    # 提取特征
    features = vgg16_model.predict(img_array)
    return features.flatten()  # 展平特征

# 使用pickle反序列化
def deserialize_features(serialized_features):
    return pickle.loads(serialized_features)

# 计算余弦相似度
def compute_similarity(query_features, db_features):
    # 计算余弦相似度
    return 1 - cdist(query_features.reshape(1, -1), db_features, metric='cosine')

@search_by_vgg_my_route.route('/searchbyvggmy', methods=['POST'])
def search_by_vgg():
    try:
        # 获取前端传递的图片文件
        file = request.files['file']
        image = np.array(cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR))

        if image is None:
            return jsonify({"error": "无法读取图像"}), 400

        # 提取查询图像的特征
        query_features = extract_vgg16_features(image)

        # 从数据库中获取所有图像的特征
        cursor.execute("SELECT id, image_path, vgg_features FROM vgg")
        rows = cursor.fetchall()

        # 反序列化数据库中的特征
        db_features = []
        image_paths = []

        for row in rows:
            image_paths.append(row[1])
            db_features.append(deserialize_features(row[2]))

        db_features = np.array(db_features)

        # 计算查询图像与数据库图像的相似度
        similarities = compute_similarity(query_features, db_features)

        # 获取最相似的图像
        most_similar_idx = np.argsort(similarities[0])[::-1]
        top_5_images = [{"image_path": image_paths[i], "similarity": float(similarities[0][i])}
                        for i in most_similar_idx[:5]]

        return jsonify({"VGG_my": top_5_images})

    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"error": str(e)}), 500

