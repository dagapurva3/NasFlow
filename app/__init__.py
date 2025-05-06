from flask import Flask
from flask_executor import Executor
from flask_socketio import SocketIO

app = Flask(__name__)
app.config.from_pyfile("../config.py")
executor = Executor(app)
socketio = SocketIO(app, async_mode="threading")

from app import routes
