import os
import cv2
import numpy as np
import mysql.connector
import json
from flask import Blueprint, request, jsonify

# 配置数据库连接
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "graduation",
    "connection_timeout": 300
}

# 创建 Flask 蓝图
search_by_color_route = Blueprint('searchbycolor', __name__)

# 提取 HSV 中心矩特征
def extract_hsv_moments(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv_image[:, :, 2]  # 亮度通道
    moments = cv2.moments(v_channel)
    hsv_moments = cv2.HuMoments(moments).flatten()
    return hsv_moments

# 提取颜色直方图特征（48 维）
def extract_color_histogram(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, [6, 6, 6], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist[:48]  # 取前 48 维

# 计算欧几里得距离
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(np.array(vec1) - np.array(vec2))

# 处理 /searchbycolor 请求
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

    hsv_results = []
    color_hist_results = []

    # 遍历数据库中的所有记录，计算相似度
    for record in records:
        db_hsv_moments = json.loads(record['hsv_moments'])
        db_color_histogram = json.loads(record['color_histogram'])

        # 计算欧几里得距离
        hsv_distance = euclidean_distance(query_hsv_moments, db_hsv_moments)
        color_hist_distance = euclidean_distance(query_color_histogram, db_color_histogram)

        # 计算相似度 (1 / (1 + 距离))，确保值越大越相似
        hsv_similarity = float(1 / (1 + hsv_distance))
        color_hist_similarity = float(1 / (1 + color_hist_distance))

        # 添加到对应的结果列表
        hsv_results.append({"image": record["image_path"], "similarity": hsv_similarity})
        color_hist_results.append({"image": record["image_path"], "similarity": color_hist_similarity})

    # 按相似度排序并取前 5 个
    hsv_results = sorted(hsv_results, key=lambda x: x["similarity"], reverse=True)[:5]
    color_hist_results = sorted(color_hist_results, key=lambda x: x["similarity"], reverse=True)[:5]

    # 关闭数据库连接
    cursor.close()
    conn.close()

    return jsonify({
        "HSV_Moments": hsv_results,
        "Color_Histogram": color_hist_results
    })