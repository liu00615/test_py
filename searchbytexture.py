import json
import cv2
import mysql.connector
import numpy as np
from flask import Blueprint, request, jsonify
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from werkzeug.utils import secure_filename
from concurrent.futures import ThreadPoolExecutor

# 创建 Flask 蓝图
search_by_texture_route = Blueprint('searchbytexture', __name__)

# 数据库配置
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "graduation",
    "connection_timeout": 300
}


# 计算相似度（归一化欧几里得距离）
def compute_similarity(query_feature, db_feature, max_distance=1000):
    # 计算欧几里得距离
    distance = np.linalg.norm(np.array(query_feature) - np.array(db_feature))
    # 将距离转换为相似度值（1 表示最相似）
    similarity = 1 - (distance / max_distance)  # 归一化到 [0,1]
    return max(0, similarity)  # 确保相似度不会小于 0


# 查询数据库
def get_database_features():
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute("SELECT id, image_path, glcm_features, lbp_features FROM texture")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


# 提取 GLCM 特征
def extract_glcm_features(image):
    image_resized = cv2.resize(image, (256, 256))  # Resize image for consistency
    gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    glcm = graycomatrix(gray_image, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], symmetric=True, normed=True)

    features = [
        graycoprops(glcm, 'contrast').mean(),
        graycoprops(glcm, 'correlation').mean(),
        graycoprops(glcm, 'energy').mean(),
        graycoprops(glcm, 'homogeneity').mean()
    ]
    return features


# 提取 LBP 特征
def extract_lbp_features(image, radius=1, n_points=8):
    image_resized = cv2.resize(image, (256, 256))  # Resize image for consistency
    gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    lbp_hist = lbp_hist.astype(np.float32)
    lbp_hist /= (lbp_hist.sum() + 1e-6)  # 归一化

    return lbp_hist.tolist()


# 计算图像相似度
def compute_image_similarity(row, query_glcm_features, query_lbp_features):
    image_id, image_path, glcm_json, lbp_json = row
    glcm_features = json.loads(glcm_json)
    lbp_features = json.loads(lbp_json)

    glcm_similarity = compute_similarity(query_glcm_features, glcm_features)
    lbp_similarity = compute_similarity(query_lbp_features, lbp_features)

    return {
        "image": image_path,
        "glcm_similarity": glcm_similarity,
        "lbp_similarity": lbp_similarity
    }


# 运行搜索
def run_search(query_image):
    query_glcm_features = extract_glcm_features(query_image)
    query_lbp_features = extract_lbp_features(query_image)

    # 获取数据库中的纹理特征
    rows = get_database_features()

    # 线程池初始化
    executor = ThreadPoolExecutor(max_workers=10)  # 设置最大线程数
    # 使用线程池并行计算每张图像的相似度
    future_to_image = {executor.submit(compute_image_similarity, row, query_glcm_features, query_lbp_features): row for
                       row in rows}

    results_glcm = []
    results_lbp = []

    for future in future_to_image:
        result = future.result()
        results_glcm.append({"image": result["image"], "similarity": result["glcm_similarity"]})
        results_lbp.append({"image": result["image"], "similarity": result["lbp_similarity"]})

    # 按照相似度排序，取前5个最相似的图像
    results_glcm = sorted(results_glcm, key=lambda x: x["similarity"], reverse=True)[:5]
    results_lbp = sorted(results_lbp, key=lambda x: x["similarity"], reverse=True)[:5]

    return {"GLCM": results_glcm, "LBP": results_lbp}


# 处理图片上传
@search_by_texture_route.route('/searchbytexture', methods=['POST'])
def search_by_texture():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 获取图像
    filename = secure_filename(file.filename)
    query_image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    if query_image is None:
        return jsonify({"error": "Invalid image file"}), 400

    # 执行图像检索
    try:
        results = run_search(query_image)
    except Exception as e:
        return jsonify({"error": f"Error during search: {str(e)}"}), 500

    return jsonify(results)
