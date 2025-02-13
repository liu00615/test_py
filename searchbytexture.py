# 使用纹理特征搜索，包括灰度共生矩阵和LBP特征
import json

import cv2
import mysql.connector
import numpy as np
from flask import Blueprint, request, jsonify
from skimage.feature import graycomatrix, graycoprops
from skimage.feature import local_binary_pattern
from werkzeug.utils import secure_filename

# 创建Flask蓝图
search_by_texture_route = Blueprint('searchbytexture', __name__)

# 数据库连接配置
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "graduation",
    "connection_timeout": 300
}

# 连接到数据库
def get_db_connection():
    return mysql.connector.connect(**db_config)


# 特征提取函数（glcm）
def extract_glcm_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    glcm = graycomatrix(gray_image, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    return [contrast, correlation, energy, homogeneity]


# 特征提取函数（lbp）
def extract_lbp_features(image, radius=1, n_points=8):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist /= (lbp_hist.sum() + 1e-6)
    return lbp_hist


# 计算相似度（欧几里得距离）
def compute_similarity(query_feature, db_feature):
    return np.linalg.norm(np.array(query_feature) - np.array(db_feature))


# 查询数据库获取图像特征
def get_database_features():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, image_path, glcm_features, lbp_features FROM texture")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


# 查询函数
def run_search(query_image):
    # 提取查询图像的纹理特征
    query_glcm_features = extract_glcm_features(query_image)
    query_lbp_features = extract_lbp_features(query_image)

    # 获取数据库中的纹理特征
    rows = get_database_features()

    # 保存结果
    results = {"glcm": [], "lbp": []}

    # 计算与数据库中每张图像的相似度
    for row in rows:
        image_id, image_path, glcm_json, lbp_json = row
        glcm_features = json.loads(glcm_json)
        lbp_features = np.array(json.loads(lbp_json))

        # 计算GLCM特征的相似度
        glcm_similarity = compute_similarity(query_glcm_features, glcm_features)
        results["glcm"].append({"image_path": image_path, "similarity": glcm_similarity})

        # 计算LBP特征的相似度
        lbp_similarity = compute_similarity(query_lbp_features, lbp_features)
        results["lbp"].append({"image_path": image_path, "similarity": lbp_similarity})

    # 按照相似度排序，取前5个最相似的图像
    results["glcm"] = sorted(results["glcm"], key=lambda x: x["similarity"])[:5]
    results["lbp"] = sorted(results["lbp"], key=lambda x: x["similarity"])[:5]

    # 返回结果
    response = {
        "glcm": results["glcm"],
        "lbp": results["lbp"]
    }

    return response


# 处理上传的图片并返回搜索结果
@search_by_texture_route.route('/searchbytexture', methods=['POST'])
def search_by_texture():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 获取图像
    filename = secure_filename(file.filename)
    img_bytes = file.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    query_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if query_image is None:
        return jsonify({"error": "Invalid image file"}), 400

    # 执行图像检索
    results = run_search(query_image)

    # 返回 JSON 格式的检索结果
    return jsonify(results)
