import signal
import subprocess
import sys

from app import app, socketio
from start_mlflow import start_mlflow_server


def main():
    # Start MLflow server
    mlflow_process = start_mlflow_server()

    try:
        # Start Flask application
        socketio.run(app, host="0.0.0.0", port=8080, debug=True, use_reloader=True)
    except KeyboardInterrupt:
        # Gracefully shutdown both servers
        mlflow_process.send_signal(signal.SIGTERM)
        mlflow_process.wait()
        sys.exit(0)


if __name__ == "__main__":
    main()
