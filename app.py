from flask import Flask, render_template, request
import os
from deepfake_model import predict_image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)

    file.save(filepath)

    result = predict_image(filepath)

    return render_template("index.html",
                           prediction=result,
                           image_path=filepath)

if __name__ == "__main__":
    app.run(debug=True)