from fastapi import FastAPI, File, UploadFile
import uvicorn  
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import tensorflow as tf
import requests

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

endpoint = "http://localhost:8502/v1/models/potatoes_model:predict"

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"

def read_file_as_image(data) -> np.ndarray:
    """
    Convert file data into a NumPy array representation of an image.
    """
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Handle prediction requests by preprocessing the input image,
    sending it to TensorFlow Serving, and returning the prediction result.
    """
    try:
        image = read_file_as_image(await file.read())
        img_batch = tf.convert_to_tensor([image], dtype=tf.float32)

        json_data = {
            "instances": img_batch.numpy().tolist()
        }

        response = requests.post(endpoint, json=json_data)

        if response.status_code != 200:
            return {
                "error": "Failed to get a valid prediction from the model server",
                "details": response.text
            }


        prediction = np.array(response.json()["predictions"][0])
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = np.max(prediction)  

        return {
            "class": predicted_class,
            "confidence": confidence
        }

    except Exception as e:
        return {
            "error": "An error occurred during prediction",
            "details": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)