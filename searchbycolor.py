import os
import cv2
import numpy as np
import mysql.connector
import json
from flask import Blueprint, request, jsonify
from scipy.spatial.distance import cosine
from numpy import corrcoef
from werkzeug.utils import secure_filename
from io import BytesIO

# 配置数据库连接
db_config = {
    "host": "localhost",
    "user": "root",  # 替换为你的 MySQL 用户名
    "password": "123456",  # 替换为你的 MySQL 密码
    "database": "graduation"
}

# 创建蓝图
search_by_color_route = Blueprint('search_by_color', __name__)


# 计算颜色直方图
def extract_color_histogram(image_data):
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        print("无法读取图像")
        return None
    image = cv2.resize(image, (256, 256))  # 统一尺寸
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()  # 归一化并展平
    return hist


# 相似度计算方法
def cosine_similarity(hist1, hist2):
    return 1 - cosine(hist1, hist2)


def euclidean_distance(hist1, hist2):
    return np.linalg.norm(hist1 - hist2)


def correlation(hist1, hist2):
    return np.corrcoef(hist1, hist2)[0, 1]


# 读取数据库中的所有图像特征
def load_database_features(cursor):
    cursor.execute("SELECT image_path, feature FROM color")
    data = cursor.fetchall()
    features = []
    for image_path, feature_json in data:
        feature_vector = np.array(json.loads(feature_json))
        features.append((image_path, feature_vector))
    return features


# 搜索函数
def search_by_color(image_data, db_features):
    test_feature = extract_color_histogram(image_data)
    if test_feature is None:
        return []

    similarities_cosine = []
    similarities_euclidean = []
    similarities_correlation = []

    for image_path, db_feature in db_features:
        cosine_sim = cosine_similarity(test_feature, db_feature)
        euclidean_sim = euclidean_distance(test_feature, db_feature)
        correlation_sim = correlation(test_feature, db_feature)

        similarities_cosine.append((image_path, cosine_sim))
        similarities_euclidean.append((image_path, euclidean_sim))
        similarities_correlation.append((image_path, correlation_sim))

    # 排序：选择与测试图片最相似的前 5 个结果
    top_5_cosine = sorted(similarities_cosine, key=lambda x: x[1], reverse=True)[:5]
    top_5_euclidean = sorted(similarities_euclidean, key=lambda x: x[1])[:5]
    top_5_correlation = sorted(similarities_correlation, key=lambda x: x[1], reverse=True)[:5]

    return top_5_cosine, top_5_euclidean, top_5_correlation


# 计算准确率
def calculate_accuracy(top_5_results, ground_truth):
    """
    计算准确率（假设前 5 个图片中，如果有正确标签则算作正确）
    ground_truth 是一个列表，包含了正确图片的路径
    """
    correct_count = sum([1 for result in top_5_results if result[0] in ground_truth])
    accuracy = correct_count / len(top_5_results) if top_5_results else 0
    return accuracy


@search_by_color_route.route('/searchbycolor', methods=['POST'])
def search():
    # 检查请求中是否有文件
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # 获取上传的文件内容
        image_data = file.read()

        # 假设真实标签的路径是通过其他方式获取，这里我们使用一个空的 ground_truth 示例
        # ground_truth 应该包含正确答案的图片路径，假设是前 5 张正确的标签图片路径
        ground_truth = ['image1.jpg', 'image3.jpg', 'image5.jpg']

        try:
            # 连接到数据库
            conn = mysql.connector.connect(**db_config)
            cursor = conn.cursor()

            # 加载数据库中的特征
            db_features = load_database_features(cursor)

            # 搜索
            top_5_cosine, top_5_euclidean, top_5_correlation = search_by_color(image_data, db_features)

            # 计算准确率
            accuracy_cosine = calculate_accuracy(top_5_cosine, ground_truth)
            accuracy_euclidean = calculate_accuracy(top_5_euclidean, ground_truth)
            accuracy_correlation = calculate_accuracy(top_5_correlation, ground_truth)

            # 返回结果
            results = {
                "cosine_similarity": {
                    "method": "Cosine Similarity",
                    "results": [{"image_path": x[0], "similarity": x[1]} for x in top_5_cosine],
                    "accuracy": accuracy_cosine
                },
                "euclidean_distance": {
                    "method": "Euclidean Distance",
                    "results": [{"image_path": x[0], "distance": x[1]} for x in top_5_euclidean],
                    "accuracy": accuracy_euclidean
                },
                "correlation": {
                    "method": "Correlation",
                    "results": [{"image_path": x[0], "correlation": x[1]} for x in top_5_correlation],
                    "accuracy": accuracy_correlation
                }
            }

            # 关闭数据库连接
            cursor.close()
            conn.close()

            return jsonify(results)

        except Exception as e:
            return jsonify({"error": str(e)}), 500
