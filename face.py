import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Load pre-trained face detection model
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_faces(image):
    """Detect faces in an image."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return image, faces

def main():
    st.title("Face Detection App")
    
    st.sidebar.title("Options")
    option = st.sidebar.radio("Choose an action:", ("Upload an Image", "Use Camera"))

    if option == "Upload an Image":
        st.header("Upload an Image")
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Detect faces
            processed_image, faces = detect_faces(image_np)

            st.subheader(f"Detected {len(faces)} face(s)")
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption="Processed Image", use_column_width=True)

    elif option == "Use Camera":
        st.header("Use Camera")
        run_camera = st.button("Start Camera")

        if run_camera:
            cap = cv2.VideoCapture(0)
            stframe = st.empty()

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.write("Failed to access camera.")
                    break

                processed_frame, faces = detect_faces(frame)

                # Convert frame for Streamlit display
                frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                stframe.image(frame_rgb, channels="RGB", use_container_width=True)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
