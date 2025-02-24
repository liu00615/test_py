import os
import pickle
import cv2
import mysql.connector

# 数据库连接配置
db_config = {
    "host": "localhost",
    "user": "root",
    "password": "123456",
    "database": "graduation",
    "connection_timeout": 3000,  # sift特征大一点，提取后插入数据库报错数据库断联，所以这里改大一点
}

# 目标数据集文件夹
base_folder = "../data/256_ObjectCategories"

# 连接数据库
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()

# 新建数据表
cursor.execute("""
    CREATE TABLE IF NOT EXISTS sift (
        id INT AUTO_INCREMENT PRIMARY KEY,
        image_path VARCHAR(255) NOT NULL,
        sift_features LONGBLOB NOT NULL  # 存储序列化的SIFT特征(以二进制形式存储)
    )
""")

# 清空sift表
cursor.execute("DELETE FROM sift;")
cursor.execute("ALTER TABLE sift AUTO_INCREMENT = 1;")
conn.commit()

print("数据库表 sift 已清空，准备插入新数据...")

# 初始化SIFT算法
sift = cv2.SIFT_create()

# 提取SIFT特征的函数
def extract_sift_features(image):
    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 检测关键点并计算描述符
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)

    return descriptors

# 每次提交的批量大小
# 一次性提取全部再插入数据库会导致数据库断开报错
batch_size = 100
count = 0

# 遍历所有子文件夹
for root, _, files in os.walk(base_folder):
    for filename in files:
        if filename.lower().endswith((".jpg", ".png", ".jpeg")):
            image_path = os.path.join(root, filename)

            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图像: {image_path}")
                continue

            # 提取SIFT特征
            sift_features = extract_sift_features(image)

            # 没有检测到特征点 跳过
            if sift_features is None:
                print(f"图像 {image_path} 没有检测到特征点，跳过...")
                continue

            # 使用pickle将SIFT特征序列化
            sift_features_pickle = pickle.dumps(sift_features)

            # 存储图像的路径和序列化的SIFT特征
            relative_path = os.path.relpath(image_path, base_folder)
            cursor.execute("""
                INSERT INTO sift (image_path, sift_features)
                VALUES (%s, %s)
            """, (relative_path, sift_features_pickle))

            count += 1
            if count % batch_size == 0:
                conn.commit()  # 每100条提交一次
                print(f"已提交 {count} 条数据...")

# 最后提交剩余的数据
conn.commit()

# 提交更改并关闭连接
cursor.close()
conn.close()

print(f"所有图像的SIFT特征已提取并存入数据库，共 {count} 条数据。")
