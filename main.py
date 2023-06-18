from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.openapi.utils import get_openapi
import joblib
import pandas as pd
import numpy as np

app = FastAPI()
model = joblib.load(open("./model/emotion_classifier_pipe_lr_18_june_2023.pkl", "rb"))
print(model)
# functions
def predict_emotions(docx):
    results =model.predict([docx])
    return results[0]
def get_prediction_proba(docx):
    results = model.predict_proba([docx])
    return results

class EmotionRequest(BaseModel):
   text: str
emotions_emoji_dict = {"anger" : "😠", "disgust" : "🤮", "fear" : "😨😱", "happy" : "🤗", "joy" : "😂", "neutral" : "😐", "sad" : "😔", "sadness" : "😔", "shame" : "😳", "surprise" : "😮"}


@app.post("/classify_emotion")
def classify_emotion(request: EmotionRequest):
 try:
    text = request.text
    # Preprocess the text if needed
    # Perform emotion classification using the loaded model
    prediction = predict_emotions(text)
    probability = get_prediction_proba(text)
    
    emoji_icon = emotions_emoji_dict[prediction]
    predictionNew = str("{}:{}".format(prediction, emoji_icon))
    probabilityNew= str("Confidence:{}".format(np.max(probability)))
    # Create the response JSON
    response = {
        "prediction": predictionNew,
        "probability": probabilityNew
    }
    return  response
 except Exception as e:
    error_msg = "An error occurred while processing the request"
    raise HTTPException(status_code=500, detail=error_msg)
        
    
@app.get("/docs", include_in_schema=False)
def custom_docs():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="Emotion Classification API")

@app.get("/openapi.json", include_in_schema=False)
def get_open_api_endpoint():
    return get_openapi(title="Emotion Classification API", version="1.0.0", routes=app.routes)
