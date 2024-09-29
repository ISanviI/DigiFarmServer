from fastapi import FastAPI, File, UploadFile
import io
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] ='1'
import uvicorn
from PIL import Image
import numpy as np
import tensorflow as tf
import joblib
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
# Installed pymon - similar to nodemon -> pymon {pythonfile.py}

load_dotenv()
PORT = int(os.getenv('PORT'))

app = FastAPI()

app.add_middleware(
  CORSMiddleware,
  allow_origins = ["https://digifarm-je7z.onrender.com/"],
  allow_credentials = True,
  allow_methods=["*"],
  allow_headers=["*"]
)

plantClasses = ['Pepper_bell__Healthy', 'Potato__Early_blight', 'Potato__Late_blight', 'Tomato__Bacterial_spot', 'Tomato__Healthy']
animalClasses = ['Lumpy_Skin', 'Normal_Skin']

cwd = os.path.dirname(os.path.abspath(__file__))
with open(f"{cwd}/models/plantModel.pkl", "rb") as p:
  plantModel = joblib.load(p)
with open(f"{cwd}/models/animalModel.pkl", "rb") as a:
  animalModel = joblib.load(a)

@app.get("/")
async def ping():
  return "Welcome to DigiFarm"

def read_file_as_image(data) -> np.ndarray:
  try:
    path = io.BytesIO(data)
    print("Data address = ", path)

    try:
      img = Image.open(io.BytesIO(data))
    except Exception as e:
      return {"error": "Invalid image file", "details": str(e)}
    
    imgResize = img.resize((256, 256))
    imgarr = np.array(imgResize)
    if (len(imgarr.shape) == 3 and imgarr.shape[2] == 3):
      img_expand = np.expand_dims(imgarr, axis=0)
    return img_expand
  
  except Exception as e:
    raise FileExistsError(f"Error reading image - {e}")
  # img_resized = tf.keras.preprocessing.image.load_img(path, target_size=(256, 256))
  # imgarr = tf.keras.preprocessing.image.img_to_array(img_resized)

@app.post("/animal")
async def predict(
  file:UploadFile = File(...)):
  try:
    imgarr = read_file_as_image(await file.read())
    predictions = animalModel.predict(imgarr)
    prediction_class = animalClasses[np.argmax(predictions[0])]
    confidence = round(100*(np.max(predictions[0])), 2)
    print("Prediction class - ", prediction_class)
    print("Confidence - ", confidence)
    return {"Prediction" : prediction_class,
            "Confidence" : confidence}
  except Exception as e:
    return {"error": "Prediction failed", "details": str(e)}

@app.post("/plant")
async def predict(file:UploadFile = File(...)):
  try:
    imgarr = read_file_as_image(await file.read())
    # if imgarr.shape != (256, 256, 3):
    #     raise ValueError(f"Image shape is incorrect. Expected (256, 256, 3), but got {imgarr.shape}")
    predictions = plantModel.predict(imgarr)
    prediction_class = plantClasses[np.argmax(predictions[0])]
    confidence = round(100*(np.max(predictions[0])), 2)
    print("Prediction class - ", prediction_class)
    print("Confidence - ", confidence)
    return {"Prediction" : prediction_class,
            "Confidence" : confidence}
  except Exception as e:
    return {"error": "Prediction failed", "details": str(e)}
  
  # content = await file.read()
  # print(content)
  # return {"message" : content}

  # Below code doesn't work
  # with open(file) as f:
  # f = open(file, 'r')
  # print(await file.read())

if __name__ == "__main__":
  uvicorn.run(app, host="localhost", port=PORT)
