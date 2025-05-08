import streamlit as st
import random
import time
from datetime import datetime
import pandas as pd
import plotly.express as px
import sqlite3
from sqlite3 import Error
import tempfile
import cv2
from ultralytics import YOLO
import torch

# Security fix for PyTorch 2.6+
torch.serialization.add_safe_globals([ultralytics.nn.tasks.DetectionModel])

# Database Setup (unchanged)
def create_connection():
    """Create a database connection"""
    conn = None
    try:
        conn = sqlite3.connect('parking.db')
        return conn
    except Error as e:
        st.error(f"Database connection error: {e}")
    return conn

# ... (keep all your existing database functions unchanged) ...

# Initialize YOLO Model (New)
@st.cache_resource
def load_yolo_model():
    try:
        model = YOLO('yolov8s.pt')  # Ensure this file exists
        return model
    except Exception as e:
        st.error(f"YOLO model loading failed: {e}")
        return None

# Video Processing Function (New)
def process_video(uploaded_file, model, lot_id):
    with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        
        # Process with YOLO
        results = model.predict(tmp.name, conf=0.5)
        
        # Count cars and update database
        car_count = len(results[0].boxes)
        update_parking_status(lot_id, car_count)
        
        # Return annotated video
        output_path = f"processed_{uploaded_file.name}"
        results[0].save(filename=output_path)
        return output_path, car_count

# Modified Admin Portal Section
def admin_portal():
    st.header("Admin Portal")
    password = st.text_input("Enter admin password", type="password")
    if password != "campus123":
        st.error("Incorrect password")
        st.stop()
    
    st.success("Admin access granted")
    tab1, tab2, tab3 = st.tabs(["Parking Status", "Analytics", "Video Processing"])  # New tab
    
    with tab1:
        # ... (keep your existing admin status UI) ...
    
    with tab2:
        # ... (keep your existing analytics UI) ...
    
    with tab3:  # New video processing tab
        st.subheader("Live Space Detection")
        selected_lot = st.selectbox(
            "Select parking lot to monitor",
            options=get_parking_lots(),
            format_func=lambda x: x['name'],
            key="video_lot"
        )
        
        uploaded_video = st.file_uploader(
            "Upload surveillance video", 
            type=["mp4", "avi"],
            key="video_upload"
        )
        
        if uploaded_video and selected_lot:
            model = load_yolo_model()
            if model:
                if st.button("Process Video"):
                    with st.spinner("Detecting vehicles..."):
                        output_path, car_count = process_video(
                            uploaded_video,
                            model,
                            selected_lot['id']
                        )
                        
                        st.success(f"Detected {car_count} vehicles")
                        st.video(output_path)
                        
                        # Update UI
                        st.rerun()

# Modified Main Function
def main():
    # ... (keep your existing setup code until view_options) ...
    
    view_options = {
        "map": "🗺️ Parking Map",
        "list": "📋 All Parking Lots", 
        "reserve": "🅿️ Reserve Spot",
        "admin": "🔒 Admin Portal",
        "monitor": "📹 Live Monitoring"  # New option
    }
    
    # ... (keep your existing navigation logic) ...
    
    # Add new monitoring view
    elif current_view == view_options["monitor"]:
        st.header("Live Parking Monitoring")
        model = load_yolo_model()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Camera Feed")
            uploaded_video = st.file_uploader(
                "Upload real-time feed", 
                type=["mp4", "avi"],
                key="live_feed"
            )
        
        with col2:
            if uploaded_video and model:
                with tempfile.NamedTemporaryFile(suffix=".mp4") as tmp:
                    tmp.write(uploaded_video.read())
                    
                    # Display processed video
                    st.subheader("Processed Feed")
                    video_placeholder = st.empty()
                    
                    cap = cv2.VideoCapture(tmp.name)
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            st.warning("End of video")
                            break
                        
                        results = model.predict(frame, conf=0.5)
                        annotated_frame = results[0].plot()
                        
                        # Convert BGR to RGB for Streamlit
                        video_placeholder.image(
                            annotated_frame[..., ::-1], 
                            channels="RGB",
                            use_column_width=True
                        )
                        
                        # Optional: Add pause button
                        if st.button("❚❚ Pause"):
                            break

if __name__ == "__main__":
    main()
