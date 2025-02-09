from flask import Flask
from searchbycolor import search_by_color_route

app = Flask(__name__)

# 注册路由
app.register_blueprint(search_by_color_route)

if __name__ == '__main__':
    app.run(debug=True)
