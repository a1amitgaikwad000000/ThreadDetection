import streamlit as st
import cv2
from ultralytics import YOLO
import pandas as pd
import time
import os

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Change to 'best.pt' if trained model

# Create folders
if not os.path.exists("screenshots"):
    os.makedirs("screenshots")

if not os.path.exists("report.csv"):
    pd.DataFrame(columns=["Time", "Object", "Confidence", "Screenshot"]).to_csv("report.csv", index=False)

st.set_page_config(page_title="Exam Surveillance", layout="centered")
st.title("🎯 AI Exam Thread Detection using YOLOv8")

run = st.checkbox("Start Camera Detection")
FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to access camera.")
            break

        results = model(frame)
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = model.names[cls_id]

                if conf > 0.5:
                    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
                    filename = f"screenshots/suspicious_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)

                    # Logging
                    df = pd.read_csv("report.csv")
                    df.loc[len(df)] = [timestamp, name, round(conf, 2), filename]
                    df.to_csv("report.csv", index=False)

                    st.error(f"⚠️ Suspicious Object: {name} ({round(conf, 2)})")

        # Show video in Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)
    
    cap.release()
else:
    st.info("Check the box to start live detection.")
