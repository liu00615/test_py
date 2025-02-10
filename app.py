from flask import Flask
from searchbycolor import search_by_color_route
from searchbyedge import search_by_edge_route
from searchbytexture import search_by_texture_route

import config  # 导入配置文件

app = Flask(__name__)

# 注册路由
app.register_blueprint(search_by_color_route)
app.register_blueprint(search_by_edge_route)
app.register_blueprint(search_by_texture_route)

if __name__ == '__main__':
    # 使用 config.py 中的 HOST 和 PORT 启动 Flask 应用
    app.run(host=config.HOST, port=config.PORT, debug=True)
