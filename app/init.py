from flask import Flask

from .mlflow_routes import mlflow_bp


def create_app():
    app = Flask(__name__)
    app.register_blueprint(mlflow_bp)
    return app


app = create_app()
