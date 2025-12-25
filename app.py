import os
import time
import cv2
import torch
import torchvision
import torch.nn as nn
from PIL import Image
import streamlit as st
from pathlib import Path
from torchvision.models import resnet18
import torchvision.transforms as transforms
import torchvision.transforms.functional as F


#Custom transformation
class ResizePad:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = F.resize(img, (new_h, new_w))

        pad_w = self.size - new_w
        pad_h = self.size - new_h
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
        return F.pad(img, [left, top, right, bottom])


#Static vatiables
BASE_DIR = Path(__file__).resolve().parent
VIDS_DIR = BASE_DIR / "Vids to test"

CUSTOM_MP = BASE_DIR / "DistractModel3.0.pth"
TRANSFER_MP = BASE_DIR / "DistractModelTransfer2.pth"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VAL_TF_CUSTOM = transforms.Compose([
    ResizePad(256),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

VAL_TF_TRANSFER = transforms.Compose([
    ResizePad(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])




#importing custom models
class CNN_Distract(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.step = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1,bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1,bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1,bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64,64,3,padding=1,bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1,bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1,bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.step(x)






#loading the custom model
@st.cache_resource #decorater to make sure you dont need to load models each time
def load_models():
   
        # Driver detector
        person_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="DEFAULT"
        ).to(DEVICE).eval()

        # Custom CNN
        custom_model = CNN_Distract(10)
        custom_model.load_state_dict(torch.load(CUSTOM_MP, map_location=DEVICE))
        custom_model.to(DEVICE).eval()

        # Transfer model
        transfer_model = resnet18()
        feats = transfer_model.fc.in_features
        transfer_model.fc = nn.Linear(feats, 10)
        transfer_model.load_state_dict(torch.load(TRANSFER_MP, map_location=DEVICE))
        transfer_model.to(DEVICE).eval()

        return person_model, custom_model, transfer_model



#Classification function
def person_present(img_rgb, model, score_thr=0.6):
    with torch.no_grad():
        tf_img = transforms.ToTensor()(img_rgb).to(DEVICE)
        out = model([tf_img])[0]
        keep = (out["scores"] >= score_thr) & (out["labels"] == 1)
        return bool(keep.sum().item())


def classify(img_rgb, model, transforms):
    with torch.no_grad():
        img = Image.fromarray(img_rgb)
        out = model(transforms(img).unsqueeze(0).to(DEVICE))
        return int(out.argmax(dim=1).item())



# =========================================================
# Streamlit UI
# =========================================================

st.set_page_config(layout="wide")
st.title("Driver Monitoring System")

with st.sidebar:
    st.header("Controls")

    video_files = sorted(
        [p for p in VIDS_DIR.iterdir() if p.suffix.lower() in {".mp4",".avi",".mov"}]
    )[:3]

    input_source = st.selectbox(
        "Input Source",
        ["Webcam"] + [v.name for v in video_files]
    )

    model_choice = st.radio(
        "Model",
        ["Custom CNN", "Transfer (ResNet18)"],
        horizontal=True
    )

    start = st.button("Start", type="primary")
    stop = st.button("Stop")

if "running" not in st.session_state:
    st.session_state.running = False

if start:
    st.session_state.running = True
if stop:
    st.session_state.running = False

video_box = st.empty()
status_box = st.empty()

person_model, custom_model, transfer_model = load_models()

cls_model = custom_model if model_choice == "Custom CNN" else transfer_model
tf = VAL_TF_CUSTOM if model_choice == "Custom CNN" else VAL_TF_TRANSFER

# Open video source
if input_source == "Webcam":
    cap = cv2.VideoCapture(0)
else:
    cap = cv2.VideoCapture(str(VIDS_DIR / input_source))

last_prediction = 4
t0 = time.time()
frames = 0

while st.session_state.running:
    ret, frame_bgr = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    if person_present(frame_rgb, person_model):
        pred = classify(frame_rgb, cls_model, tf)
        label = f"Class {pred}"
        last_prediction = pred
    else:
        label = "NO DRIVER"
        last_prediction = 4

    cv2.putText(frame_bgr, label, (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    
    
    video_box.image(
        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB),
        width='stretch'
    )

    frames += 1
    fps = frames / max(time.time() - t0, 1e-6)

    status_box.markdown(
        f"""
        **Status**
        - Input: `{input_source}`
        - Model: `{model_choice}`
        - Prediction: `{label}`
        - FPS: `{fps:.1f}`
        - Device: `{DEVICE}`
        """
    )
    time.sleep(1.5)


cap.release()
