from fastapi import FastAPI, File, UploadFile
import uvicorn  
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import tensorflow as tf
from keras.layers import TFSMLayer

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

# Load the model using TFSMLayer
MODEL = TFSMLayer("../saved_models/1", call_endpoint="serving_default")
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

@app.get("/ping")
async def ping():
    return "Hello, I am alive"


def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess the image
    image = read_file_as_image(await file.read())
    img_batch = tf.convert_to_tensor([image], dtype=tf.float32)

    # Call the TFSMLayer
    predictions = MODEL(img_batch)

    if isinstance(predictions, dict):
        predictions = next(iter(predictions.values()))  

    # Debugging information
    # print("Predictions:", predictions)
    # print("Shape of predictions:", predictions.shape)

    predicted_class = CLASS_NAMES[tf.argmax(predictions[0]).numpy()]
    confidence = tf.reduce_max(predictions[0]).numpy()
    
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)
