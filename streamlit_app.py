import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from datetime import datetime
from ultralytics import YOLO
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="UZ Smart Parking",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .header {
        color: #FFD700;
        font-family: 'Arial', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #002366;
    }
    .stButton>button {
        background-color: #FFD700;
        color: #002366;
        font-weight: bold;
    }
    .stSelectbox, .stTextInput, .stNumberInput {
        background-color: #f0f2f6;
    }
    .stExpander {
        background-color: #f0f2f6;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "parking_lots" not in st.session_state:
    st.session_state.parking_lots = [
        {"id": 1, "name": "Main Car Park", "capacity": 50, "occupied": 0, "reserved": []},
        {"id": 2, "name": "Library Parking", "capacity": 30, "occupied": 0, "reserved": []},
        {"id": 3, "name": "Great Hall Parking", "capacity": 20, "occupied": 0, "reserved": []}
    ]

if "auth" not in st.session_state:
    st.session_state.auth = {
        "logged_in": False,
        "role": None,
        "username": None,
        "department": None
    }

# University credentials
UZ_CREDENTIALS = {
    "admin": {"password": "uzadmin2024", "role": "admin", "department": "Administration"},
    "student123": {"password": "uzstudent", "role": "student", "department": "Mathematics"},
    "lecturer456": {"password": "uzlecturer", "role": "lecturer", "department": "Mathematics"}
}

PURPOSE_OF_VISIT = ["Lecture Attendance", "Business Meeting", "Ceremony/Event", "Research", "Other"]

# Sidebar login
with st.sidebar:
    st.image("https://www.centralprocurement.com/wp-content/uploads/2023/04/University-of-Zimbabwe-Logo-2.jpg", width=120)
    st.title("University of Zimbabwe")
    st.subheader("Smart Parking System")
    
    if st.session_state.auth["logged_in"]:
        st.success(f"Welcome {st.session_state.auth['role'].title()}")
        st.write(f"Department: {st.session_state.auth['department']}")
        if st.button("Logout"):
            st.session_state.auth = {"logged_in": False, "role": None, "username": None, "department": None}
            st.rerun()
    else:
        st.subheader("Login")
        username = st.text_input("Staff/Student ID")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if username in UZ_CREDENTIALS and UZ_CREDENTIALS[username]["password"] == password:
                st.session_state.auth = {
                    "logged_in": True,
                    "role": UZ_CREDENTIALS[username]["role"],
                    "username": username,
                    "department": UZ_CREDENTIALS[username]["department"]
                }
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")

# Load YOLO model
@st.cache_resource(show_spinner="Loading parking detection model...")
def load_model():
    try:
        model = YOLO('yolov8n.pt')
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

def process_video(video_path, model, lot_id):
    cap = cv2.VideoCapture(video_path)
    car_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame, verbose=False)
        for result in results:
            boxes = result.boxes
            car_count += len([box for box in boxes if box.cls == 2])
    
    cap.release()
    
    lot = next(lot for lot in st.session_state.parking_lots if lot["id"] == lot_id)
    lot["occupied"] = min(car_count, lot["capacity"])
    
    return car_count

def reserve_spot(lot_id, user_id, duration_minutes, purpose):
    lot = next(lot for lot in st.session_state.parking_lots if lot["id"] == lot_id)
    
    if len(lot["reserved"]) >= lot["capacity"] - lot["occupied"]:
        return False
        
    lot["reserved"].append({
        "user_id": user_id,
        "start_time": datetime.now(),
        "duration": duration_minutes,
        "purpose": purpose,
        "department": st.session_state.auth["department"],
        "paid": False
    })
    return True

# User Dashboard
def user_dashboard():
    st.title("🏛️ UZ Smart Parking")
    st.subheader(f"Welcome {st.session_state.auth['role'].title()} from {st.session_state.auth['department']}")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("Available Parking Lots")
        for lot in st.session_state.parking_lots:
            with st.expander(f"📍 {lot['name']} - {lot['capacity'] - lot['occupied'] - len(lot['reserved'])} spots available"):
                purpose = st.selectbox("Purpose of Visit", PURPOSE_OF_VISIT, key=f"purpose_{lot['id']}")
                duration = st.number_input("Duration (minutes)", 
                                         min_value=30, 
                                         max_value=240, 
                                         value=60,
                                         key=f"dur_{lot['id']}")
                
                if st.button(f"Reserve Spot at {lot['name']}", key=f"res_{lot['id']}"):
                    if reserve_spot(lot["id"], st.session_state.auth["username"], duration, purpose):
                        st.success(f"Reservation confirmed for {purpose}!")
                    else:
                        st.error("No available spots. Please try another lot.")
                    st.rerun()

    with col2:
        st.header("Quick Stats")
        total_capacity = sum(lot["capacity"] for lot in st.session_state.parking_lots)
        total_occupied = sum(lot["occupied"] + len(lot["reserved"]) for lot in st.session_state.parking_lots)
        
        st.metric("Total Spaces", total_capacity)
        st.metric("Available Spaces", total_capacity - total_occupied)
        
        # Pie chart of purposes
        purposes = [res["purpose"] for lot in st.session_state.parking_lots for res in lot["reserved"]]
        if purposes:
            purpose_df = pd.DataFrame({"Purpose": purposes})
            fig = px.pie(purpose_df, names="Purpose", title="Parking Purposes")
            st.plotly_chart(fig, use_container_width=True)

# Admin Dashboard
def admin_dashboard():
    st.title("🛠️ UZ Parking Administration")
    
    tab1, tab2, tab3 = st.tabs(["📹 Live Monitoring", "📋 Reservations", "📊 Analytics"])

    with tab1:
        st.header("CCTV Parking Monitoring")
        uploaded_video = st.file_uploader("Upload CCTV footage", type=["mp4", "mov"])
        selected_lot = st.selectbox("Select Parking Lot", 
                                  st.session_state.parking_lots, 
                                  format_func=lambda x: x["name"])
        
        model = load_model()
        
        if uploaded_video and model and st.button("Analyze Parking"):
            with st.spinner("Detecting vehicles..."):
                temp_video = "temp_video.mp4"
                with open(temp_video, "wb") as f:
                    f.write(uploaded_video.getbuffer())
                
                car_count = process_video(temp_video, model, selected_lot["id"])
                st.success(f"Detected {car_count} vehicles in {selected_lot['name']}")
                st.video(temp_video)
                os.remove(temp_video)
                st.rerun()

    with tab2:
        st.header("Current Reservations")
        reservation_data = []
        for lot in st.session_state.parking_lots:
            for res in lot["reserved"]:
                reservation_data.append({
                    "Lot": lot["name"],
                    "User": res["user_id"],
                    "Department": res["department"],
                    "Purpose": res["purpose"],
                    "Duration": f"{res['duration']} mins",
                    "Time": res["start_time"].strftime("%Y-%m-%d %H:%M")
                })
        
        if reservation_data:
            st.dataframe(pd.DataFrame(reservation_data), 
                        use_container_width=True,
                        hide_index=True)
        else:
            st.info("No current reservations")

    with tab3:
        st.header("Parking Analytics")
        
        # Capacity utilization
        utilization_data = []
        for lot in st.session_state.parking_lots:
            utilization = (lot["occupied"] + len(lot["reserved"])) / lot["capacity"] * 100
            utilization_data.append({
                "Lot": lot["name"],
                "Capacity": lot["capacity"],
                "Occupied": lot["occupied"] + len(lot["reserved"]),
                "Utilization": utilization
            })
        
        df = pd.DataFrame(utilization_data)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Lot Utilization")
            fig = px.bar(df, x="Lot", y="Utilization", color="Lot",
                        title="Parking Lot Utilization (%)")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Purpose Distribution")
            purposes = [res["purpose"] for lot in st.session_state.parking_lots for res in lot["reserved"]]
            if purposes:
                purpose_df = pd.DataFrame({"Purpose": purposes})
                fig = px.pie(purpose_df, names="Purpose", 
                            title="Purpose of Visits")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No reservation data available")

# Main App Logic
def main():
    if not st.session_state.auth["logged_in"]:
        st.title("🏛️ University of Zimbabwe")
        st.subheader("Smart Parking Management System")
        st.image("https://th.bing.com/th/id/OIP.W5g71oYhOBS_PSlzQtNLCwHaHa?w=193&h=193&c=7&r=0&o=5&dpr=1.5&pid=1.7", width=300)
        st.markdown("""
        <div style="text-align: center; margin-top: 10px;">
            <h4>Please login from the sidebar to access parking services</h4>
            <p>For assistance, contact: parking@uz.ac.zw</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        if st.session_state.auth["role"] == "admin":
            admin_dashboard()
        else:
            user_dashboard()

if __name__ == "__main__":
    main()
