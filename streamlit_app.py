import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
from datetime import datetime
import os
from ultralytics import YOLO

# Set page config - MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Smart Parking System",
    page_icon="🚗",
    layout="wide"
)

# Initialize session state for parking data
if "parking_lots" not in st.session_state:
    st.session_state.parking_lots = [
        {"id": 1, "name": "Lot A", "capacity": 20, "occupied": 0, "reserved": []},
        {"id": 2, "name": "Lot B", "capacity": 30, "occupied": 0, "reserved": []}
    ]

# Load YOLO model with caching
@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model():
    try:
        # Use smaller nano model for better performance
        model = YOLO('yolov8n.pt')
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

def process_video(video_path, model, lot_id):
    cap = cv2.VideoCapture(video_path)
    car_count = 0
    
    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run detection
        results = model(frame, verbose=False)
        
        # Count cars (class 2 in COCO dataset)
        for result in results:
            boxes = result.boxes
            car_count += len([box for box in boxes if box.cls == 2])
    
    cap.release()
    
    # Update parking lot status
    lot = next(lot for lot in st.session_state.parking_lots if lot["id"] == lot_id)
    lot["occupied"] = min(car_count, lot["capacity"])
    
    return car_count

def reserve_spot(lot_id, user_id, duration_minutes):
    lot = next(lot for lot in st.session_state.parking_lots if lot["id"] == lot_id)
    
    if len(lot["reserved"]) >= lot["capacity"] - lot["occupied"]:
        return False
        
    lot["reserved"].append({
        "user_id": user_id,
        "start_time": datetime.now(),
        "duration": duration_minutes,
        "paid": False
    })
    return True

def admin_portal():
    st.title("🚗 Smart Parking Management System")
    
    # Password protection
    password = st.text_input("Admin Password", type="password")
    if password != "admin123":
        st.error("Incorrect password")
        st.stop()
    
    st.success("✅ Admin access granted")
    
    tab1, tab2, tab3 = st.tabs(["Live Monitoring", "Reservations", "Analytics"])

    with tab1:
        st.header("🖥️ Live Parking Monitoring")
        uploaded_video = st.file_uploader("Upload CCTV Feed", type=["mp4", "mov"])
        selected_lot = st.selectbox(
            "Select Parking Lot",
            st.session_state.parking_lots,
            format_func=lambda x: x["name"]
        )
        
        model = load_model()
        
        if uploaded_video and model and st.button("Analyze Parking"):
            with st.spinner("Processing video..."):
                temp_video = "temp_video.mp4"
                with open(temp_video, "wb") as f:
                    f.write(uploaded_video.getbuffer())
                
                car_count = process_video(temp_video, model, selected_lot["id"])
                st.success(f"Detected {car_count} vehicles in {selected_lot['name']}")
                st.video(temp_video)
                os.remove(temp_video)
                st.rerun()

    with tab2:
        st.header("📅 Manage Reservations")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("New Reservation")
            lot = st.selectbox(
                "Parking Lot", 
                st.session_state.parking_lots,
                format_func=lambda x: x["name"],
                key="reserve_lot"
            )
            user_id = st.text_input("User ID")
            duration = st.number_input("Duration (minutes)", min_value=15, max_value=1440)
            
            if st.button("Reserve Spot"):
                if reserve_spot(lot["id"], user_id, duration):
                    st.success("Reservation confirmed!")
                else:
                    st.error("No available spots")
                st.rerun()
        
        with col2:
            st.subheader("Current Reservations")
            for lot in st.session_state.parking_lots:
                with st.expander(f"{lot['name']} ({len(lot['reserved'])} reserved)"):
                    for res in lot["reserved"]:
                        st.write(f"User {res['user_id']} - {res['duration']} mins")

    with tab3:
        st.header("📊 Parking Analytics")
        df = pd.DataFrame(st.session_state.parking_lots)
        df["utilization"] = (df["occupied"] / df["capacity"]) * 100
        
        st.subheader("Occupancy Rates")
        st.bar_chart(df.set_index("name")[["occupied", "capacity"]])
        
        st.subheader("Key Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Capacity", sum(lot["capacity"] for lot in st.session_state.parking_lots))
        with col2:
            total_utilization = df["occupied"].sum() / df["capacity"].sum() * 100
            st.metric("Overall Utilization", f"{total_utilization:.1f}%")

def main():
    try:
        admin_portal()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
