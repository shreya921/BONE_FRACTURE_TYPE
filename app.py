# app.py
import os
import gdown

MODEL_PATH = "cnn_model.keras"

if not os.path.exists(MODEL_PATH):
    file_id = "1DeMxMip6IEsOesnjs93W-yMLasTYaXlh"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)
import numpy as np
import pandas as pd
import joblib
import gradio as gr
from PIL import Image
from tensorflow.keras.models import load_model

# ---------- Load Models ----------
cnn_model = None
xgb_model = joblib.load("xgb_model.pkl")

bone_types = [
    "clavicle","femur","finger","foot","forearm","hand","humerus","humerus shaft",
    "pelvis","radius","shoulder","tibia","ulna","wrist","ankle","elbow"
]

class_names = [
    "Distal Fracture",
    "Non Fracture",
    "Post Fracture",
    "Proximal Fracture"
]

def preprocess_image(image):
    image = image.convert("RGB").resize((224,224))
    img = np.array(image)/255.0
    img = np.expand_dims(img,0)
    return img

def preprocess_metadata(age, gender, left_right, bone_width,
                        fracture_gap, gap_visibility, bone_type):
    gender = 1 if gender=="Male" else 0
    left_right = 1 if left_right=="Right" else 0
    gap_map={"No":0,"Slight":1,"Yes":2}
    gap_visibility=gap_map[gap_visibility]
    bone_vector=[0]*16
    if bone_type in bone_types:
        bone_vector[bone_types.index(bone_type)] = 1
    metadata=[age,gender,left_right,bone_width,fracture_gap,gap_visibility]
    metadata.extend(bone_vector)
    return np.array(metadata).reshape(1,-1)

def predict(image,age,gender,left_right,bone_width,fracture_gap,gap_visibility,bone_type):
    global cnn_model

    if cnn_model is None:
    print("Loading CNN model...")
    cnn_model = load_model(MODEL_PATH)
    img=preprocess_image(image)
    meta=preprocess_metadata(age,gender,left_right,bone_width,fracture_gap,gap_visibility,bone_type)
    cnn_prob=cnn_model.predict(img,verbose=0)
    xgb_prob=xgb_model.predict_proba(meta)
    fusion_prob=0.5*cnn_prob+0.5*xgb_prob
    pred=np.argmax(fusion_prob)
    conf=float(np.max(fusion_prob))*100
    probs={class_names[i]:float(fusion_prob[0][i]) for i in range(len(class_names))}
    return class_names[pred],f"{conf:.2f}%",probs

with open("style.css", "r") as f:
    css = f.read()

with gr.Blocks(theme=gr.themes.Soft(),css=css,title="OrthoVision AI") as demo:
    gr.HTML("""
    <div class='hero'>
    <h1>🦴 OrthoVision AI</h1>
    <h3>AI Powered Bone Fracture Detection</h3>
    <p>Upload an X-ray and patient metadata for AI-assisted fracture analysis.</p>
    </div>
    """)
    with gr.Row():
        with gr.Column():
            image=gr.Image(type="pil",label="Upload X-ray")
            age=gr.Number(label="Age")
            gender=gr.Dropdown(["Male","Female"],label="Gender")
            left_right=gr.Dropdown(["Left","Right"],label="Affected Side")
        with gr.Column():
            bone_width=gr.Number(label="Bone Width")
            fracture_gap=gr.Number(label="Fracture Gap")
            gap_visibility=gr.Dropdown(["No","Slight","Yes"],label="Gap Visibility")
            bone_type=gr.Dropdown(bone_types,label="Bone Type")
            btn=gr.Button("Analyze",variant="primary")
    pred=gr.Textbox(label="Prediction")
    conf=gr.Textbox(label="Confidence")
    probs=gr.Label(label="Class Probabilities")
    btn.click(
        predict,
        [image,age,gender,left_right,bone_width,fracture_gap,gap_visibility,bone_type],
        [pred,conf,probs]
    )

if __name__=="__main__":
    demo.launch(server_name="0.0.0.0",
                server_port=int(os.environ.get("PORT",7860)))
