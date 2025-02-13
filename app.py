from flask import Flask
from flask_cors import CORS

from searchbycolor import search_by_color_route
from searchbyedge import search_by_edge_route
from searchbytexture import search_by_texture_route
from searchbysift import search_by_sift_route
from searchbyvgg import search_by_vgg_route
from searchbyvgg_keras import search_by_vgg_keras_route
from searchbyhash import search_by_hash_route

import config  # 导入配置文件

app = Flask(__name__)

# 配置 CORS 前端跨域问题
CORS(app)

# 注册路由
app.register_blueprint(search_by_color_route)
app.register_blueprint(search_by_edge_route)
app.register_blueprint(search_by_texture_route)
app.register_blueprint(search_by_sift_route)
app.register_blueprint(search_by_vgg_route)
app.register_blueprint(search_by_vgg_keras_route)
app.register_blueprint(search_by_hash_route)

if __name__ == '__main__':
    # config.py用于配置后端服务器的ip地址和监听端口
    app.run(host=config.HOST, port=config.PORT, debug=True)
