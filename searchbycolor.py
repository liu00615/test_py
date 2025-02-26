import json
import os

import cv2
import mysql.connector
import numpy as np
from flask import Blueprint, request, jsonify
from concurrent.futures import ThreadPoolExecutor

# 配置数据库连接
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "graduation",
    "connection_timeout": 300
}

# 创建Flask蓝图
search_by_color_route = Blueprint('searchbycolor', __name__)

# 预处理步骤：图像缩放、去噪和直方图均衡化
def preprocess_image(image, target_size=(256, 256)):
    image = cv2.resize(image, target_size)  # 缩放图像
    image = cv2.GaussianBlur(image, (5, 5), 0)  # 高斯滤波去噪
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[:, :, 2] = cv2.equalizeHist(hsv_image[:, :, 2])  # 亮度通道直方图均衡化
    return hsv_image

# 提取HSV中心矩特征
def extract_hsv_moments(image):
    hsv_image = preprocess_image(image)
    v_channel = hsv_image[:, :, 2]  # 亮度通道
    moments = cv2.moments(v_channel)
    hsv_moments = cv2.HuMoments(moments).flatten()
    return hsv_moments

# 提取颜色直方图特征（48 维）
def extract_color_histogram(image):
    hsv_image = preprocess_image(image)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [6, 6, 6], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist[:48]  # 取前48维

# 计算欧几里得距离
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(np.array(vec1) - np.array(vec2))

# 计算相似度并返回结果
def compute_similarity(record, query_hsv_moments, query_color_histogram):
    db_hsv_moments = json.loads(record['hsv_moments'])
    db_color_histogram = json.loads(record['color_histogram'])

    # 计算欧几里得距离
    hsv_distance = euclidean_distance(query_hsv_moments, db_hsv_moments)
    color_hist_distance = euclidean_distance(query_color_histogram, db_color_histogram)

    # 计算相似度 (1 / (1 + 距离))，值越大越相似
    hsv_similarity = float(1 / (1 + hsv_distance))
    color_hist_similarity = float(1 / (1 + color_hist_distance))

    return {
        "image": record["image_path"],
        "similarity": hsv_similarity
    }, {
        "image": record["image_path"],
        "similarity": color_hist_similarity
    }

# /searchbycolor请求
@search_by_color_route.route('/searchbycolor', methods=['POST'])
def search_by_color():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty file name"}), 400

    # 读取上传的图像
    image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({"error": "Invalid image"}), 400

    # 提取特征
    query_hsv_moments = extract_hsv_moments(image)
    query_color_histogram = extract_color_histogram(image)

    # 连接数据库
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor(dictionary=True)

    # 获取数据库中所有记录
    cursor.execute("SELECT image_path, hsv_moments, color_histogram FROM color")
    records = cursor.fetchall()

    # 使用线程池并行计算相似度
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(compute_similarity, record, query_hsv_moments, query_color_histogram) for record in records]
        results = [future.result() for future in futures]

    # 处理并按相似度排序
    hsv_results = [result[0] for result in results]
    color_hist_results = [result[1] for result in results]

    # 按相似度排序并取前5个
    hsv_results = sorted(hsv_results, key=lambda x: x["similarity"], reverse=True)[:5]
    color_hist_results = sorted(color_hist_results, key=lambda x: x["similarity"], reverse=True)[:5]

    # 关闭数据库连接
    cursor.close()
    conn.close()

    return jsonify({
        "HSV_Moments": hsv_results,
        "Color_Histogram": color_hist_results
    })
