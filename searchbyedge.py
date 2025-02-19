from flask import Blueprint, request, jsonify
import cv2
import numpy as np
import json
import mysql.connector
from sklearn.metrics.pairwise import cosine_similarity

# 配置数据库连接
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "graduation"
}

# 连接数据库
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

# 定义Flask蓝图
search_by_edge_route = Blueprint('searchbyedge', __name__)

# 提取Hu不变矩特征
def extract_hu_moments(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    moments = cv2.moments(contours[0])
    hu_moments = cv2.HuMoments(moments).flatten()
    return hu_moments

# 提取HOG特征（降维，只取前 500 维）
def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(gray, (128, 128))  # 降低计算复杂度
    hog = cv2.HOGDescriptor()
    hog_features = hog.compute(image)
    if hog_features is not None:
        hog_features = hog_features.flatten()
        # 降维，只取前500维
        if len(hog_features) > 500:
            hog_features = hog_features[:500]
    return hog_features

# 计算余弦相似度
def compute_similarity(query_feature, db_feature):
    return cosine_similarity([query_feature], [db_feature])[0][0]

# 处理前端请求并进行图像检索
@search_by_edge_route.route('/searchbyedge', methods=['POST'])
def search_by_edge():
    # 获取前端上传的图片，name="file"
    file = request.files['file']
    query_image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    if query_image is None:
        return jsonify({"error": "无法读取图像"}), 400

    # 提取查询图片的Hu不变矩和HOG特征
    query_hu_moments = extract_hu_moments(query_image)
    query_hog_features = extract_hog_features(query_image)

    # 从数据库获取所有图像的边缘特征
    cursor.execute("SELECT id, image_path, hu_moments, hog_features FROM edge")
    rows = cursor.fetchall()

    hu_results = []  # 存储基于 Hu 不变矩的搜索结果
    hog_results = []  # 存储基于 HOG 特征的搜索结果

    for row in rows:
        db_id, image_path, hu_json, hog_json = row
        db_hu_moments = np.array(json.loads(hu_json))
        db_hog_features = np.array(json.loads(hog_json))

        # 计算Hu不变矩的相似度
        hu_similarity = compute_similarity(query_hu_moments, db_hu_moments)

        # 计算HOG特征的相似度
        hog_similarity = compute_similarity(query_hog_features, db_hog_features)

        # Hu不变矩结果
        hu_results.append({
            "image_path": image_path,
            "similarity": hu_similarity
        })

        # HOG特征结果
        hog_results.append({
            "image_path": image_path,
            "similarity": hog_similarity
        })

    # 按照相似度排序，返回前五个最相似的结果
    hu_results = sorted(hu_results, key=lambda x: x['similarity'], reverse=True)[:5]
    hog_results = sorted(hog_results, key=lambda x: x['similarity'], reverse=True)[:5]

    # 构造最终返回的结果
    response = {
        "Hu_Moments": [{"image": result['image_path'], "similarity": result['similarity']} for result in hu_results],
        "HOG_Features": [{"image": result['image_path'], "similarity": result['similarity']} for result in hog_results]
    }

    # 返回最相似的五个图像及其相似度
    return jsonify(response)
