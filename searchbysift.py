import pickle
import cv2
import mysql.connector
import numpy as np
from flask import Blueprint, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor

# 创建Flask蓝图
search_by_sift_route = Blueprint('searchbysift', __name__)

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

# 线程池
executor = ThreadPoolExecutor(max_workers=10)

# SIFT特征提取函数
def extract_sift_features(image):
    sift = cv2.SIFT_create()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray_image, None)
    return des

# 计算相似度的函数
def calculate_similarity(query_features, sift_features, img_id, img_path):
    # 使用余弦相似度比较
    similarity = cosine_similarity(query_features, sift_features)
    return (img_id, img_path, similarity[0][0])

# 搜索函数，基于SIFT特征
@search_by_sift_route.route('/searchbysift', methods=['POST'])
def search_by_sift():
    # 获取前端上传的查询图片
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 读取图片
    query_image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if query_image is None:
        return jsonify({"error": "Unable to read the image"}), 400

    # 提取查询图片的SIFT特征
    query_features = extract_sift_features(query_image)
    if query_features is None:
        return jsonify({"error": "No keypoints found in the query image"}), 400

    # 从数据库中获取所有图像的SIFT特征
    cursor.execute("SELECT id, image_path, sift_features FROM sift")
    all_images = cursor.fetchall()

    # 并行计算每张图片与查询图片的相似度
    futures = []
    for img in all_images:
        img_id, img_path, sift_feature_blob = img

        # 使用pickle反序列化二进制数据
        sift_features = pickle.loads(sift_feature_blob)

        if sift_features is None:
            continue  # 跳过没有SIFT特征的图像

        # 将任务提交给线程池
        futures.append(executor.submit(calculate_similarity, query_features, sift_features, img_id, img_path))

    # 获取所有线程的结果
    similarities = [future.result() for future in futures]

    # 按照相似度排序，取前5张最相似的图片
    similarities.sort(key=lambda x: x[2], reverse=True)
    top_5_results = similarities[:5]

    results = {
        "SIFT": [
            {"image": img_path, "similarity": float(similarity)}
            for _, img_path, similarity in top_5_results
        ]
    }

    # 返回JSON格式的结果
    return jsonify(results)
