from flask import Blueprint, jsonify

# Define the Blueprint
api_blueprint = Blueprint('api', __name__)

# Example Route 1: Health Check
@api_blueprint.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok", "message": "API is running"}), 200

# Example Route 2: A Test Endpoint
@api_blueprint.route('/hello', methods=['GET'])
def say_hello():
    return jsonify({"message": "Hello from the API Blueprint!"}), 200
