import json
import re
from urllib.parse import urlencode, urljoin, urlparse

import requests
from flask import Blueprint, Response, request

mlflow_bp = Blueprint("mlflow", __name__, url_prefix="/mlflow")
MLFLOW_SERVER = "http://localhost:3001"


def rewrite_content(content):
    content = content.replace(MLFLOW_SERVER, "/mlflow")
    content = re.sub(r'(href|src|action)=([\'"])(\/[^/])', r"\1=\2/mlflow/\3", content)
    return content


@mlflow_bp.route("/", defaults={"path": ""})
@mlflow_bp.route("/<path:path>")
def proxy(path):
    # Build target URL with query parameters
    target_url = urljoin(MLFLOW_SERVER, path)
    if request.query_string:
        target_url = f"{target_url}?{request.query_string.decode('utf-8')}"

    # Forward headers
    headers = {
        key: value
        for (key, value) in request.headers
        if key.lower() not in ["host", "content-length"]
    }
    headers["Host"] = urlparse(MLFLOW_SERVER).hostname

    # Get request data
    data = request.get_data()
    if request.is_json:
        headers["Content-Type"] = "application/json"

    # Make proxy request
    try:
        resp = requests.request(
            method=request.method,
            url=target_url,
            headers=headers,
            data=data,
            cookies=request.cookies,
            allow_redirects=False,
            stream=True,
        )
    except requests.RequestException as e:
        return Response(str(e), status=500)

    # Process response
    excluded_headers = [
        "content-encoding",
        "content-length",
        "transfer-encoding",
        "connection",
        "x-frame-options",  # Remove X-Frame-Options to allow embedding
    ]
    headers = [
        (name, value)
        for (name, value) in resp.raw.headers.items()
        if name.lower() not in excluded_headers
    ]

    # Add header to allow embedding in iframe
    headers.append(("X-Frame-Options", "SAMEORIGIN"))

    # Handle different content types
    content = resp.content
    content_type = resp.headers.get("Content-Type", "")

    if content_type.startswith("text/html"):
        content = rewrite_content(content.decode("utf-8")).encode("utf-8")
    elif content_type.startswith("application/json"):
        # Ensure proper JSON handling
        try:
            json_content = json.loads(content)
            return Response(
                json.dumps(json_content),
                status=resp.status_code,
                headers=headers,
                content_type="application/json",
            )
        except json.JSONDecodeError:
            pass

    return Response(content, resp.status_code, headers)
