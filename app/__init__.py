from flask import Flask
from flask_executor import Executor
from flask_socketio import SocketIO

from .mlflow_routes import mlflow_bp  # Add this import

app = Flask(__name__)
app.config.from_pyfile("../config.py")
app.config['TEMPLATES_AUTO_RELOAD'] = False  # Prevent template auto-reloading
executor = Executor(app)
socketio = SocketIO(app, async_mode='threading', ping_timeout=60, ping_interval=25)

# Register the MLflow blueprint
app.register_blueprint(mlflow_bp)

from app import routes
