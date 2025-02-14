import pickle
import mysql.connector
import numpy as np
from PIL import Image
from flask import Blueprint, request, jsonify
from scipy.fftpack import dct

# 定义Flask蓝图
search_by_hash_route = Blueprint('searchbyhash', __name__)

# 数据库连接配置
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "graduation",
    "connection_timeout": 300
}

# 哈希方法的计算函数
def average_hash(img, hash_size=8):
    img = img.convert('L')
    img = img.resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(img)
    avg = pixels.mean()
    diff = pixels > avg
    return diff.flatten()

def difference_hash(img, hash_size=9):
    img = img.convert('L')
    img = img.resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(img)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return diff.flatten()

def perceptual_hash(img, hash_size=32):
    img = img.convert('L')  # 转为灰度图像
    img = img.resize((hash_size, hash_size), Image.Resampling.LANCZOS)  # 调整图像大小
    pixels = np.array(img)  # 转换为NumPy数组

    # 进行DCT变换
    dct_result = dct(pixels, axis=0)  # 沿着列方向进行DCT
    dct_result = dct(dct_result, axis=1)  # 沿着行方向进行DCT

    # 获取低频部分（8x8块）
    dct_low_freq = dct_result[:8, :8]

    # 计算平均值，并根据平均值生成哈希值
    avg = dct_low_freq.mean()
    diff = dct_low_freq > avg  # 高于均值的部分标记为 1，低于均值的部分为 0

    return diff.flatten()  # 展开为一维数组

# 计算哈希距离
def hamming_distance(hash1, hash2):
    return np.count_nonzero(hash1 != hash2)

# 计算相似度
def calculate_similarity(distance, max_distance):
    return 1 - (distance / max_distance)

# 加载哈希特征的数据库表
def load_hash_features(cursor):
    cursor.execute("SELECT id, image_path, aHash, dHash, pHash FROM hash")
    rows = cursor.fetchall()
    return rows

# 根据查询图像返回最相似的图像列表
def search_similar_images(query_hash, method, cursor):
    rows = load_hash_features(cursor)
    results = []

    for row in rows:
        image_id, image_path, a_hash, d_hash, p_hash = row
        if method == 'aHash':
            hash_to_compare = pickle.loads(a_hash)
        elif method == 'dHash':
            hash_to_compare = pickle.loads(d_hash)
        elif method == 'pHash':
            hash_to_compare = pickle.loads(p_hash)
        else:
            continue

        distance = hamming_distance(query_hash, hash_to_compare)
        max_distance = len(query_hash)  # 哈希值的最大距离就是哈希长度
        similarity = calculate_similarity(distance, max_distance)
        results.append((image_id, image_path, similarity))

    # 按照相似度排序，越高表示越相似
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:5]

# 处理前端请求
@search_by_hash_route.route('/searchbyhash', methods=['POST'])
def search_by_hash():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # 打开上传的文件作为图像对象
    img = Image.open(file)

    # 提取三种哈希特征
    a_hash_query = average_hash(img)
    d_hash_query = difference_hash(img)
    p_hash_query = perceptual_hash(img)

    # 连接数据库
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()

    # 获取搜索结果
    a_hash_results = search_similar_images(a_hash_query, 'aHash', cursor)
    d_hash_results = search_similar_images(d_hash_query, 'dHash', cursor)
    p_hash_results = search_similar_images(p_hash_query, 'pHash', cursor)

    # 关闭数据库连接
    cursor.close()
    conn.close()

    # 构造返回结果
    response = {
        "aHash": [{"image": result[1], "similarity": result[2]} for result in a_hash_results],
        "dHash": [{"image": result[1], "similarity": result[2]} for result in d_hash_results],
        "pHash": [{"image": result[1], "similarity": result[2]} for result in p_hash_results]
    }

    return jsonify(response)
