from fastapi import FastAPI
from fastapi.datastructures import UploadFile
from fastapi.param_functions import File
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import requests

app = FastAPI()


#Docker terminal command
#docker run -t --rm -p 8501:8501 -v /Users/kazmi/Stuff/tomato_disease:/tomato_disease emacski/tensorflow-serving --rest_api_port=8501 --model_config_file=tomato_disease/models.config

endpoint = "http://localhost:8501/v1/models/tomatoes_model:predict"


#MODEL.load_weights('Saved_weights/1.h5') 
CLASS_NAMES = ['Tomato_Bacterial_spot',
 'Tomato_Early_blight',
 'Tomato_Late_blight',
 'Tomato_Leaf_Mold',
 'Tomato_Septoria_leaf_spot',
 'Tomato_Spider_mites_Two_spotted_spider_mite',
 'Tomato__Target_Spot',
 'Tomato__Tomato_YellowLeaf__Curl_Virus',
 'Tomato__Tomato_mosaic_virus',
 'Tomato_healthy']

# Jupyter NB part ends

@app.get("/ping")
async def ping():
    return "Hello, This server is working. yes!"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):
    image = read_file_as_image(await file.read())
    image_batch = np.expand_dims(image, 0)
    json_data = {
        "instances": image_batch.tolist()
    }
    response = requests.post(endpoint, json=json_data)
    prediction = np.array(response.json()["predictions"][0])
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction)
    return {
        "class": predicted_class,
        "confidence": float(confidence), 
    }


if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)   