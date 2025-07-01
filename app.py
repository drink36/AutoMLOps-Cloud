
import os

from flask import Flask, request
from werkzeug.middleware.proxy_fix import ProxyFix

from inference import model_fn, input_fn, predict_fn, output_fn


app = Flask(__name__)

# Load the model by reading the `SM_MODEL_DIR` environment variable
# which is passed to the container by SageMaker (usually /opt/ml/model).
model = model_fn(os.environ["SM_MODEL_DIR"])

# Since the web application runs behind a proxy (nginx), we need to
# add this setting to our app.
app.wsgi_app = ProxyFix(
    app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1
)


@app.route("/ping", methods=["GET"])
def ping():
    """
    Healthcheck function.
    """
    return "pong"


@app.route("/invocations", methods=["POST"])
def invocations():

    content_type = request.headers.get("Content-Type", "application/json")
    body = request.data 

    parsed_input = input_fn(body, content_type)
    if parsed_input.empty:
        return "", 204

    prediction = predict_fn(parsed_input, model)

    accept = request.headers.get("Accept", "application/csv")
    result = output_fn(prediction, accept)

    return result, 200, {"Content-Type": accept}
