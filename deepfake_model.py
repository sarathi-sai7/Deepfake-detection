import numpy as np
import cv2
from tensorflow.keras.models import load_model

IMG_SIZE = 224

model = load_model("xception_model.h5")

print("Xception model loaded successfully")

def predict_image(image_path):

    img = cv2.imread(image_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)[0][0]

    confidence = prediction * 100

    print("Prediction value:", prediction)

    if prediction >= 0.5:
        return f"Real ({confidence:.2f}%)"
    else:
        return f"Fake ({100-confidence:.2f}%)"