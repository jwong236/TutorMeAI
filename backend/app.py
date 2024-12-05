from flask import Flask
from app.routes.routes import api_blueprint

# Initialize the Flask app
app = Flask(__name__)

# Register Blueprints
app.register_blueprint(api_blueprint, url_prefix='/api')

# Basic Configuration
app.config['DEBUG'] = True  # Enable debug mode during development

# Default Route
@app.route('/')
def home():
    return "Welcome to the Flask Backend!"

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
