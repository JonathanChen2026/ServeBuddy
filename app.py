import streamlit as st
import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import tempfile
import os
import pickle
from collections import deque
import imageio
from PIL import Image


# Set page configuration
st.set_page_config(page_title='Tennis Serve Analyzer', layout='wide')

# Load the trained model
@st.cache_resource
def load_model():
    with open('final.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

model = load_model()

# Mapping from class names to descriptive issue names
issue_descriptions = {
    'GoodServe': 'Good Serve',
    'Issue1': 'Lack of Knee Bend',
    'Issue2': 'Too Much Forward Motion',
    'Issue3': 'Poor Footwork',
    'Issue4': 'Swing Motion Too Compact'
}


def extract_landmarks_and_overlay(video_path, output_path='processed_video.mp4', n=2, slow_factor=3):
    """
    Processes a video file and extracts pose landmarks using Mediapipe.
    Draws pose landmarks on each frame and saves the video with overlays.
    Returns a DataFrame containing the landmarks.
    """
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    landmarks_list = []
    
    # Prepare video writer
    fps = cap.get(cv2.CAP_PROP_FPS) / slow_factor  # Reduce fps to create a slow-motion effect
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = imageio.get_writer(output_path, fps=fps, format='FFMPEG')

    # Initialize variables for serve detection
    movement_window_size = 7
    movement_smoothing_window = 10
    min_serve_duration = 30
    max_serve_duration = 170
    movement_threshold_multiplier = 1.7
    peak_movement_fraction = 0.3
    
    movement_history = deque(maxlen=movement_window_size)
    movement_smoothing_history = deque(maxlen=movement_smoothing_window)
    movement_threshold = None
    serve_detected = False
    serving_started = False
    serve_frame_counter = 0
    peak_movement = 0
    prev_landmarks = None
    
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_instance:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Exit if no frames left to process

            frame_count += 1

            # Process every nth frame
            if frame_count % n != 0:
                continue  # Skip this frame

            # Recolor image to RGB for Mediapipe
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False  # Improve performance

            # Make detection
            results = pose_instance.process(image)

            # Extract pose landmarks
            try:
                current_landmarks = results.pose_landmarks.landmark

                # Draw landmarks on the frame
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(
                    image, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),  # Landmark points
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=4, circle_radius=2)   # Connections
                )

                # Convert back to RGB for correct colors in the saved video
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Save the processed frame multiple times to create slow-motion effect
                for _ in range(slow_factor):
                    writer.append_data(image_rgb)

                # Serve detection logic (same as before)
                # Select key landmarks
                key_indices = [11, 12, 13, 14, 15, 16, 23, 24]  # Shoulders, elbows, wrists, hips
                key_landmarks = [current_landmarks[i] for i in key_indices]
                
                if prev_landmarks is not None:
                    # Compute movement magnitude
                    movement = 0
                    for prev_lm, curr_lm in zip(prev_landmarks, key_landmarks):
                        dx = curr_lm.x - prev_lm.x
                        dy = curr_lm.y - prev_lm.y
                        dz = curr_lm.z - prev_lm.z
                        dist = np.sqrt(dx**2 + dy**2 + dz**2)
                        movement += dist
                    
                    # Append to movement history
                    movement_history.append(movement)
                    
                    # Smooth the movement signal
                    movement_smoothing_history.append(movement)
                    smoothed_movement = np.mean(movement_smoothing_history)
                    
                    # Calculate dynamic movement threshold if not set
                    if movement_threshold is None and frame_count > movement_window_size:
                        baseline_movement = np.mean(list(movement_history))
                        movement_threshold = baseline_movement * movement_threshold_multiplier
                    
                    # Serve detection logic
                    if not serve_detected and movement_threshold is not None:
                        if smoothed_movement > movement_threshold:
                            serving_started = True
                            serve_detected = True
                            serve_frame_counter = 0
                            peak_movement = smoothed_movement
                    elif serving_started:
                        serve_frame_counter += 1
                        
                        # Update peak movement
                        if smoothed_movement > peak_movement:
                            peak_movement = smoothed_movement
                        
                        # Check for serve end conditions
                        if (smoothed_movement < peak_movement * peak_movement_fraction and
                            serve_frame_counter >= min_serve_duration) or \
                            serve_frame_counter >= max_serve_duration:
                            serving_started = False
                else:
                    # Initialize movement history
                    movement_history.append(0)
                    movement_smoothing_history.append(0)
                
                # Update previous landmarks
                prev_landmarks = key_landmarks
                
                # Extract landmarks if serving_started is True
                if serving_started:
                    # Flatten the landmark coordinates
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] 
                                              for landmark in current_landmarks]).flatten())
                    landmarks_list.append(pose_row)
            except Exception as e:
                pass  # Skip frame if landmarks are not detected

        cap.release()
        writer.close()  # Close the video writer

    # Convert landmarks_list to DataFrame
    if landmarks_list:
        num_coords = len(results.pose_landmarks.landmark)
        columns = []
        for val in range(1, num_coords + 1):
            columns += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
        df = pd.DataFrame(landmarks_list, columns=columns)
        return df
    else:
        return None




def preprocess_landmarks(df):
    """
    Preprocesses the landmarks DataFrame for prediction.
    """
    if df is not None:
        y_cols = [col for col in df.columns if col.startswith('y')]
        nose_y = df['y1']  # Assuming 'y1' corresponds to the nose
        for col in y_cols:
            df[col] = df[col] / nose_y
    return df

def aggregate_landmarks(df):
    """
    Aggregates frame-level landmarks into a single feature vector.
    """
    if df is not None and not df.empty:
        aggregated_df = df.mean(axis=0).to_frame().T
        return aggregated_df
    else:
        return None



# App title
st.title('ServeBuddy 🎾 - Your Personal Serve Analyzer')

st.markdown("""
### Instructions:
1. 🎥 **Upload or Record** your tennis serve video.
2. 🧐 **Analyze** to get insights on potential issues.
3. 💡 **Get Tips** on how to improve your serve.
""")



# Sidebar options
st.sidebar.title('Options')
app_mode = st.sidebar.selectbox('Choose the mode', ['Upload Video', 'Record Video'])

if app_mode == 'Upload Video':
    uploaded_file = st.file_uploader('Upload a video file', type=['mp4', 'mov', 'avi'])

    if uploaded_file is not None:
        # Save uploaded video to a temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        # Process the video
        with st.spinner('Analyzing the video...'):
            output_video_path = 'processed_video.mp4'
            df_landmarks = extract_landmarks_and_overlay(tfile.name, output_path=output_video_path)

        # Create columns for side-by-side layout
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Video")
            st.video(tfile.name)  # Display the original video

        if df_landmarks is not None:
            with col2:
                st.subheader("Serve Analysis with Pose Overlay")
                st.video(output_video_path)  # Display the processed video with pose overlay

            # Preprocess, aggregate, and predict
            df_preprocessed = preprocess_landmarks(df_landmarks)
            df_aggregated = aggregate_landmarks(df_preprocessed)
            prediction = model.predict(df_aggregated)
            prediction_proba = model.predict_proba(df_aggregated)

            st.success(f'The model predicts: **{prediction[0]}**')

            class_names = model.classes_

            # Display probabilities with colored progress bars
            st.subheader('Prediction Probabilities:')
            colors = ['#4CAF50', '#FFC107', '#2196F3', '#FF5722', '#9C27B0']  # Customize as needed

            for idx, class_name in enumerate(class_names):
                probability = prediction_proba[0][idx]
                issue_name = issue_descriptions.get(class_name, class_name)
                st.write(f'{issue_name}: {probability:.2%}')
                # Custom progress bar using HTML and CSS
                progress_bar = f"""
                <div style='background-color: lightgray; border-radius: 5px; width: 100%;'>
                    <div style='width: {probability * 100}%; background-color: {colors[idx]}; height: 20px; border-radius: 5px;'></div>
                </div>
                """
                st.markdown(progress_bar, unsafe_allow_html=True)

            # Expanders with tips and embedded YouTube videos
            st.subheader('Detailed Feedback')
            for idx, class_name in enumerate(class_names):
                issue_name = issue_descriptions.get(class_name, class_name)
                with st.expander(f'{issue_name}'):
                    st.write('Tips for improvement:')
                    if class_name == 'Issue1':
                        st.write('- Bend your knees more during your serve.')
                        st.video('https://www.youtube.com/watch?v=example1')
                    elif class_name == 'Issue2':
                        st.write('- Avoid excessive forward motion.')
                        st.video('https://www.youtube.com/watch?v=example2')
                    # Add tips and videos for other classes

        else:
            st.error('No landmarks were detected in the video.')




# elif app_mode == 'Record Video':
#     st.write('Click the button below to start recording your serve using your webcam.')

#     record = st.button('Start Recording')

#     if record:
#         st.warning('Recording will start in 5 seconds. Please get ready.')
#         import time
#         time.sleep(5)
#         st.info('Recording... Press "q" to stop.')

#         # Initialize webcam
#         cap = cv2.VideoCapture(1)
#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#         # Define the codec and create VideoWriter object
#         fourcc = cv2.VideoWriter_fourcc(*'XVID')
#         out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height))

#         while cap.isOpened():
#             ret, frame = cap.read()
#             if ret:
#                 # Write the frame to the output file
#                 out.write(frame)

#                 # Show the recording status
#                 st.image(frame, channels="BGR")

#                 # Break on 'q' key press
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
#             else:
#                 break

#         # Release resources
#         cap.release()
#         out.release()
#         cv2.destroyAllWindows()

#         st.success('Recording complete.')

#         # Process the recorded video
#         with st.spinner('Analyzing the recorded video...'):
#             df_landmarks = extract_landmarks_from_video('output.avi')

#         if df_landmarks is not None:
#             # Preprocess, aggregate, and predict
#             df_preprocessed = preprocess_landmarks(df_landmarks)
#             df_aggregated = aggregate_landmarks(df_preprocessed)
#             prediction = model.predict(df_aggregated)
#             prediction_proba = model.predict_proba(df_aggregated)

#             predicted_issue = issue_descriptions.get(prediction[0], prediction[0])
#             st.success(f'The model predicts: **{predicted_issue}**')

#             class_names = model.classes_

#             # Display probabilities with colored progress bars
#             st.subheader('Prediction Probabilities:')
#             colors = ['#4CAF50', '#FFC107', '#2196F3', '#FF5722', '#9C27B0']  # Customize as needed

#             for idx, class_name in enumerate(class_names):
#                 probability = prediction_proba[0][idx]
#                 issue_name = issue_descriptions.get(class_name, class_name)
#                 st.write(f'{issue_name}: {probability:.2%}')
#                 # Custom progress bar using HTML and CSS
#                 progress_bar = f"""
#                 <div style='background-color: lightgray; border-radius: 5px; width: 100%;'>
#                     <div style='width: {probability * 100}%; background-color: {colors[idx]}; height: 20px; border-radius: 5px;'></div>
#                 </div>
#                 """
#                 st.markdown(progress_bar, unsafe_allow_html=True)

#             # Expanders with tips and embedded YouTube videos
#             st.subheader('Detailed Feedback')
#             for idx, class_name in enumerate(class_names):
#                 issue_name = issue_descriptions.get(class_name, class_name)
#                 with st.expander(f'{issue_name}'):
#                     st.write('Tips for improvement:')
#                     if class_name == 'Issue1':
#                         st.write('- Bend your knees more during your serve.')
#                         st.video('https://www.youtube.com/watch?v=example1', width=400, height=225)
#                     elif class_name == 'Issue2':
#                         st.write('- Avoid excessive forward motion.')
#                         st.video('https://www.youtube.com/watch?v=example2', width=400, height=225)
#                     # Add tips and videos for other classes
#         else:
#             st.error('No landmarks were detected in the video.')
