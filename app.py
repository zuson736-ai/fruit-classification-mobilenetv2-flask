from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

app = Flask(__name__)

# ======================
# LOAD MODEL
# ======================
model = load_model("model/fruit_model_final.keras")
model.make_predict_function()

# ======================
# LOAD LABEL
# ======================
with open("model/labels.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# ======================
# FOLDER UPLOAD
# ======================
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ======================
# PREDICT FUNCTION
# ======================


def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    label = class_names[class_index]
    confidence = float(np.max(prediction)) * 100

    return label, confidence

# ======================
# ROUTE HOME
# ======================


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            label, confidence = predict_image(filepath)

            image_path = filepath

            result = label

    return render_template(
        "index.html",
        result=result,
        confidence=confidence,
        image_path=image_path
    )


# ======================
# RUN APP
# ======================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
