import signal
import subprocess
import sys
import logging

from app import app, socketio
from start_mlflow import start_mlflow_server

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting NasFlow application")
    
    # Start MLflow server
    logger.info("Starting MLflow server")
    mlflow_process = start_mlflow_server()

    try:
        # Start Flask application
        logger.info("Starting Flask-SocketIO server")
        socketio.run(app, host="0.0.0.0", port=8080, debug=True, use_reloader=True)
    except KeyboardInterrupt:
        # Gracefully shutdown both servers
        logger.info("Shutting down servers")
        mlflow_process.send_signal(signal.SIGTERM)
        mlflow_process.wait()
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        mlflow_process.send_signal(signal.SIGTERM)
        mlflow_process.wait()
        sys.exit(1)


if __name__ == "__main__":
    main()
