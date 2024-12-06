from flask import Flask
from app.routes.routes import api_blueprint

app = Flask(__name__)

app.register_blueprint(api_blueprint, url_prefix='/api')

app.config['DEBUG'] = True

@app.route('/')
def home():
    return "Welcome to the Flask Backend!"

if __name__ == '__main__':
    app.run()
