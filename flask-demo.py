from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return 'Hello, World!'

# ---

from flask import request
from markupsafe import escape

@app.route("/hello")
def hello():
    name = request.args.get("name", "Flask")
    return f"Hello, {escape(name)}!"

# ---





# flask --app flask-demo run