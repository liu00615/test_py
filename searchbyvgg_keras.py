import io
import pickle

import mysql.connector
import numpy as np
from PIL import Image
from flask import Blueprint, request, jsonify
from keras.src.applications.vgg16 import VGG16
from tensorflow.keras import Model
from tensorflow.keras import layers

# 创建 Flask 蓝图
search_by_vgg_keras_route = Blueprint('searchbyvggkeras', __name__)

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

# 加载预训练的VGG16模型
def build_vgg16(input_shape=(224, 224, 3)):
    """
    加载 VGG16 模型，去掉顶层全连接层，用于特征提取。
    """
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    # 提取卷积层的输出作为特征
    x = layers.GlobalAveragePooling2D()(base_model.output)
    feature_model = Model(inputs=base_model.input, outputs=x)
    return feature_model

# 加载模型
vgg16_model = build_vgg16()

# 提取图像的特征
def extract_vgg16_features(img):
    # 提取传入图像的 VGG16 特征
    img = img.resize((224, 224))  # 调整为模型输入大小
    img_array = np.array(img)  # 转换为数组
    img_array = np.expand_dims(img_array, axis=0)  # 增加批次维度
    img_array = img_array / 255.0  # 归一化

    # 提取特征
    features = vgg16_model.predict(img_array)
    return features.flatten()  # 展平特征

# 搜索接口
@search_by_vgg_keras_route.route('/searchbyvggkeras', methods=['POST'])
def search_by_vgg_keras():
    # 获取上传的图像
    file = request.files['file']

    # 将图像转换为PIL图像对象
    img = Image.open(io.BytesIO(file.read()))

    # 提取查询图像的特征
    query_features = extract_vgg16_features(img)

    # 查找数据库中最相似的图像
    cursor.execute("SELECT id, image_path, vgg_features FROM vgg_keras")
    rows = cursor.fetchall()

    # 计算余弦相似度
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    results = []
    for row in rows:
        db_id, image_path, vgg_features_serialized = row
        # 反序列化特征
        vgg_features = pickle.loads(vgg_features_serialized)

        # 计算相似度
        similarity = cosine_similarity(query_features, vgg_features)

        # 将similarity转换为Python float
        similarity = float(similarity)

        results.append({
            'image': image_path,
            'similarity': similarity
        })

    # 排序结果，按相似度从高到低
    results = sorted(results, key=lambda x: x['similarity'], reverse=True)

    # 返回前5个最相似的图像，并以符合要求的格式返回
    return jsonify({
        "VGG16": [
            {"image": result['image'], "similarity": result['similarity']}
            for result in results[:5]
        ]
    })
