import os
import signal
import subprocess
import sys
import time

import mlflow


def start_mlflow_server():
    # Convert Windows path to proper format
    tracking_uri = os.path.abspath("mlruns").replace("\\", "/")
    db_path = os.path.join(tracking_uri, "mlflow.db")
    
    # Verify the database exists
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"MLflow database not found at {db_path}")
    
    # Start the MLflow server with the existing sqlite database

    process = subprocess.Popen([
        "mlflow", "server",
        "--host", "127.0.0.1",
        "--port", "3001",
        "--backend-store-uri", f"sqlite:///{db_path}",
        "--default-artifact-root", f"{tracking_uri}"
    ])
    
    print(f"MLflow server started with database: {db_path}")
    print(f"MLflow UI available at: http://127.0.0.1:3001")
    
    # Wait for server to start
    time.sleep(1)
    
    return process

if __name__ == "__main__":
    mlflow_process = start_mlflow_server()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down MLflow server...")
        mlflow_process.send_signal(signal.SIGTERM)
        mlflow_process.wait()
        sys.exit(0)