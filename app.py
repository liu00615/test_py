from flask import Flask
from searchbycolor import search_by_color_route
from searchbyedge import search_by_edge_route
from searchbytexture import search_by_texture_route
from searchbysift import search_by_sift_route
from searchbyvgg import search_by_vgg_route
from searchbyvgg_keras import search_by_vgg_keras_route

import config  # 导入配置文件

app = Flask(__name__)

# 注册路由
app.register_blueprint(search_by_color_route)
app.register_blueprint(search_by_edge_route)
app.register_blueprint(search_by_texture_route)
app.register_blueprint(search_by_sift_route)
app.register_blueprint(search_by_vgg_route)
app.register_blueprint(search_by_vgg_keras_route)

if __name__ == '__main__':
    # 使用 config.py 中的 HOST 和 PORT 启动 Flask 应用
    app.run(host=config.HOST, port=config.PORT, debug=True)
