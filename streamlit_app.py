import cv2
import numpy as np
from ultralytics import YOLO
import streamlit as st
import tempfile
from datetime import datetime, timedelta
import hashlib

# Database simulation (in production, use real DB)
users_db = {
    "admin": {
        "password": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": "admin",
        "name": "System Admin"
    },
    "user1": {
        "password": hashlib.sha256("user123".encode()).hexdigest(),
        "role": "user",
        "name": "John Doe",
        "license_plate": "ABC123"
    }
}

parking_db = {
    "spaces": [False]*12,
    "reservations": {},
    "history": []
}

# Initialize session state
if 'auth' not in st.session_state:
    st.session_state.auth = None
if 'parking_data' not in st.session_state:
    st.session_state.parking_data = parking_db.copy()

# Security functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate(username, password):
    if username in users_db:
        if users_db[username]['password'] == hash_password(password):
            return users_db[username]
    return None

# Load model and classes (cached)
@st.cache_resource
def load_model():
    model = YOLO('yolov8s.pt')
    with open("coco.txt", "r") as f:
        class_list = f.read().split("\n")
    return model, class_list

model, class_list = load_model()

# Parking space coordinates (Faculty of Science)
parking_areas = [
    [(52,364),(30,417),(73,412),(88,369)],   # Space 1
    [(105,353),(86,428),(137,427),(146,358)], # Space 2
    # ... add all 12 areas
]

# Core functions
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        st.error("Couldn't read video frame")
        return False
    
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    counts = [0] * 12
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls)
            if class_list[cls] == 'car':
                cx, cy = (x1+x2)//2, (y1+y2)//2
                for i, area in enumerate(parking_areas):
                    if cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False) >= 0:
                        counts[i] += 1
    
    # Update parking data
    occupied = sum(min(1, count) for count in counts)
    st.session_state.parking_data['spaces'] = [count > 0 for count in counts]
    st.session_state.parking_data['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return True

def reserve_space(space_number, user):
    if st.session_state.parking_data['spaces'][space_number-1]:
        st.error(f"Space {space_number} is already occupied!")
        return
    
    with st.form(key=f'reserve_form_{space_number}'):
        st.subheader(f"Reserve Space {space_number}")
        duration = st.selectbox("Duration", ["30 minutes", "1 hour", "2 hours", "4 hours"])
        
        if st.form_submit_button("Confirm Reservation"):
            end_time = datetime.now() + timedelta(
                hours=int(duration.split()[0]) if "hour" in duration else timedelta(minutes=30))
            
            reservation = {
                'space': space_number,
                'user': user['name'],
                'username': st.session_state.auth['username'],
                'license_plate': user.get('license_plate', 'N/A'),
                'reserved_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'duration': duration,
                'expires_at': end_time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            st.session_state.parking_data['reservations'][space_number] = reservation
            st.session_state.parking_data['spaces'][space_number-1] = True
            st.session_state.parking_data['history'].append(reservation)
            st.success(f"Space {space_number} reserved successfully!")
            st.rerun()

# UI Components
def login_section():
    st.title("Faculty Parking System Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            user = authenticate(username, password)
            if user:
                st.session_state.auth = {
                    "username": username,
                    **user
                }
                st.rerun()
            else:
                st.error("Invalid credentials")

def parking_availability():
    st.header("📊 Parking Availability")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Spaces", 12)
    free = 12 - sum(st.session_state.parking_data['spaces'])
    col2.metric("Free Spaces", free)
    col3.metric("Occupied Spaces", 12 - free)
    
    st.caption(f"Last updated: {st.session_state.parking_data.get('last_updated', 'Never')}")
    
    # Parking grid
    st.subheader("Parking Spaces - Faculty of Science")
    cols = st.columns(4)
    
    for i in range(12):
        space_number = i + 1
        with cols[i%4]:
            container = st.container(border=True)
            occupied = st.session_state.parking_data['spaces'][i]
            
            if occupied:
                if space_number in st.session_state.parking_data['reservations']:
                    res = st.session_state.parking_data['reservations'][space_number]
                    container.error(f"📌 Reserved by {res['user']}")
                    container.caption(f"Plate: {res['license_plate']}")
                    container.caption(f"Until: {res['expires_at']}")
                else:
                    container.error("🔴 Occupied")
            else:
                container.success("🟢 Available")
                if container.button(f"Reserve {space_number}", key=f"reserve_{space_number}"):
                    reserve_space(space_number, st.session_state.auth)

def reservation_history():
    st.header("🕒 Reservation History")
    
    if not st.session_state.parking_data['history']:
        st.info("No reservation history found")
        return
    
    history = sorted(
        st.session_state.parking_data['history'],
        key=lambda x: x['reserved_at'],
        reverse=True
    )
    
    if st.session_state.auth['role'] == 'admin':
        df = pd.DataFrame(history)
        st.dataframe(df)
    else:
        user_history = [h for h in history if h['username'] == st.session_state.auth['username']]
        if not user_history:
            st.info("You have no past reservations")
            return
        
        for res in user_history:
            with st.container(border=True):
                cols = st.columns([1,3])
                cols[0].subheader(f"Space {res['space']}")
                cols[1].write(f"📅 {res['reserved_at']}")
                cols[1].write(f"⏱️ {res['duration']}")
                cols[1].write(f"🛑 {res['expires_at']}")

def admin_panel():
    st.header("🔧 Admin Panel")
    
    tab1, tab2 = st.tabs(["Upload Surveillance", "Manage System"])
    
    with tab1:
        st.subheader("Update Parking Status")
        uploaded_file = st.file_uploader("Upload latest parking lot video", type=["mp4", "mov"])
        
        if uploaded_file and st.button("Process Video"):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            
            with st.spinner("Analyzing parking spaces..."):
                if process_video(tfile.name):
                    st.success("Parking status updated!")
                else:
                    st.error("Processing failed")
    
    with tab2:
        st.subheader("System Management")
        st.write(f"Logged in as: {st.session_state.auth['name']} (Admin)")
        st.write(f"Total reservations: {len(st.session_state.parking_data['history'])}")
#                                                                           ^ Added this parenthesis
        
        if st.button("Clear All Reservations"):
            st.session_state.parking_data['reservations'] = {}
            st.session_state.parking_data['spaces'] = [False]*12
            st.success("All reservations cleared")
            st.rerun()

def main_app():
    st.sidebar.title("Navigation")
    
    if st.session_state.auth['role'] == 'admin':
        menu_options = {
            "Availability": parking_availability,
            "Reservations": reservation_history,
            "Admin Panel": admin_panel
        }
    else:
        menu_options = {
            "Availability": parking_availability,
            "Reservations": reservation_history
        }
    
    selected = st.sidebar.radio(
        "Menu",
        list(menu_options.keys()),
        label_visibility="collapsed"
    )
    
    st.sidebar.divider()
    st.sidebar.write(f"Logged in as: {st.session_state.auth['name']}")
    if st.sidebar.button("Logout"):
        st.session_state.auth = None
        st.session_state.parking_data = parking_db.copy()
        st.rerun()
    
    menu_options[selected]()

def main():
    st.set_page_config(
        page_title="Faculty Parking System",
        page_icon="🚗",
        layout="wide"
    )
    
    if not st.session_state.auth:
        login_section()
    else:
        main_app()

if __name__ == "__main__":
    main()
