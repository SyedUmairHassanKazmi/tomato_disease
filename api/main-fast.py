from fastapi import FastAPI
from fastapi.datastructures import UploadFile
from fastapi.param_functions import File
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import load_model

app = FastAPI()


# Jupyter Notebook Part


image_shape = (224,224)
input_shape = image_shape + (3,)



base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                include_top=False, 
                                                weights='imagenet')


base_model.trainable = False

inputs = tf.keras.Input(shape=input_shape) 

x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

x = base_model(x,training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x) 
x = tf.keras.layers.Dropout(0.2)(x)

outputs = tf.keras.layers.Dense(10, activation = 'softmax')(x)

MODEL = tf.keras.Model(inputs, outputs)

model2 = MODEL.layers[3]
model2.trainable = True
fine_tune_at = 130
for layer in model2.layers[:fine_tune_at]:
    layer.trainable = False

MODEL.load_weights('saved_w/1.h5')

CLASS_NAMES = CLASS_NAMES = ['Tomato_Bacterial_spot',
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
    image = np.resize(image, (224,224,3))
    image_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(image_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class' : predicted_class,
        'confidence' : float(confidence),
    }

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)   