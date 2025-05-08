def admin_portal():
    st.header("Admin Portal")
    password = st.text_input("Enter admin password", type="password")
    if password != "campus123":
        st.error("Incorrect password")
        st.stop()
    
    st.success("Admin access granted")
    tab1, tab2, tab3 = st.tabs(["Parking Status", "Analytics", "Video Processing"])
    
    with tab1:  # Parking Status
        st.subheader("Manage Parking Lots")
        for lot in parking_lots:
            with st.expander(lot['name']):
                new_occupied = st.number_input(
                    "Occupied spots",
                    min_value=0,
                    max_value=lot['capacity'],
                    value=lot['occupied'],
                    key=f"admin_{lot['id']}"
                )
                if st.button(f"Update {lot['name']}", key=f"update_{lot['id']}"):
                    if update_parking_status(lot['id'], new_occupied):
                        st.success("Updated successfully!")
                        time.sleep(0.5)
                        st.rerun()
    
    with tab2:  # Analytics
        st.subheader("Parking Analytics")
        data = []
        for lot in parking_lots:
            data.append({
                "Lot": lot['name'],
                "Capacity": lot['capacity'],
                "Occupied": lot['occupied'],
                "Utilization": lot['occupied'] / lot['capacity'] * 100
            })
        
        df = pd.DataFrame(data)
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(df, x="Lot", y=["Capacity", "Occupied"])
        with col2:
            st.metric("Total Campus Capacity", sum(lot['capacity'] for lot in parking_lots))
            st.metric("Current Utilization", 
                     f"{sum(lot['occupied'] for lot in parking_lots)/sum(lot['capacity'] for lot in parking_lots)*100:.1f}%")
    
    with tab3:  # Video Processing
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
            if model and st.button("Process Video"):
                with st.spinner("Detecting vehicles..."):
                    output_path, car_count = process_video(
                        uploaded_video,
                        model,
                        selected_lot['id']
                    )
                    st.success(f"Detected {car_count} vehicles")
                    st.video(output_path)
                    st.rerun()
