import json
import numpy as np
import cv2
from flask import Blueprint, request, jsonify
import mysql.connector
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics.pairwise import cosine_similarity
import time

from sklearn.preprocessing import StandardScaler

# 配置数据库连接
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "graduation"
}

# 连接数据库
def get_db_connection():
    try:
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        return conn, cursor
    except mysql.connector.Error as e:
        print(f"数据库连接错误: {e}")
        return None, None


# 定义Flask蓝图
search_by_edge_route = Blueprint('searchbyedge', __name__)


# 图像预处理：去噪、归一化和图像缩放
def preprocess_image(image, target_size=(128, 128)):
    # 高斯滤波去噪
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # 将图像缩放到目标尺寸
    image = cv2.resize(image, target_size)

    return image


# 提取Hu不变矩特征
def extract_hu_moments(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    moments = cv2.moments(contours[0])
    hu_moments = cv2.HuMoments(moments).flatten()

    # 归一化Hu Moments
    hu_moments = np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-6)  # 防止对数为负无穷
    # 标准化Hu Moments
    scaler = StandardScaler()
    hu_moments = scaler.fit_transform(hu_moments.reshape(-1, 1)).flatten()

    return hu_moments


# 提取HOG特征（降维，只取前 500 维）
def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = preprocess_image(gray)  # 预处理图像
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

# 计算欧氏距离作为相似度
def compute_similarity_o(query_feature, db_feature):
    return (np.linalg.norm(query_feature - db_feature)/10)*3.5


# 处理单个数据库记录的相似度计算
def process_record(record, query_hu_moments, query_hog_features):
    db_id, image_path, hu_json, hog_json = record
    db_hu_moments = np.array(json.loads(hu_json))
    db_hog_features = np.array(json.loads(hog_json))

    # 计算Hu不变矩的相似度
    hu_similarity = compute_similarity_o(query_hu_moments, db_hu_moments)

    # 计算HOG特征的相似度
    hog_similarity = compute_similarity(query_hog_features, db_hog_features)

    return {
        "image_path": image_path,
        "hu_similarity": hu_similarity,
        "hog_similarity": hog_similarity
    }


# 处理前端请求并进行图像检索
@search_by_edge_route.route('/searchbyedge', methods=['POST'])
def search_by_edge():
    start_time = time.time()

    # 获取前端上传的图片，name="file"
    file = request.files['file']
    query_image = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    if query_image is None:
        return jsonify({"error": "无法读取图像"}), 400

    # 提取查询图片的Hu不变矩和HOG特征
    query_hu_moments = extract_hu_moments(query_image)
    query_hog_features = extract_hog_features(query_image)

    # 从数据库获取所有图像的边缘特征（限制一次读取的行数，避免加载过多数据）
    conn, cursor = get_db_connection()
    if conn is None:
        return jsonify({"error": "数据库连接失败"}), 500

    cursor.execute("SELECT id, image_path, hu_moments, hog_features FROM edge LIMIT 100")  # 仅获取前100条记录
    rows = cursor.fetchall()

    hu_results = []  # 存储基于 Hu 不变矩的搜索结果
    hog_results = []  # 存储基于 HOG 特征的搜索结果

    # 使用线程池并行处理相似度计算
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for row in rows:
            futures.append(executor.submit(process_record, row, query_hu_moments, query_hog_features))

        for future in as_completed(futures):
            result = future.result()
            hu_results.append({
                "image_path": result["image_path"],
                "similarity": result["hu_similarity"]
            })
            hog_results.append({
                "image_path": result["image_path"],
                "similarity": result["hog_similarity"]
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
