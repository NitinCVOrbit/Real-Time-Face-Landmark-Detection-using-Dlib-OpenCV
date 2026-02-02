import cv2                    
import dlib                    # Dlib library for face detection and landmarks
from imutils import face_utils # Utility functions for handling facial landmarks

# ---------------- Load Models ----------------

# Load dlib's pre-trained face detector 
detector = dlib.get_frontal_face_detector()

# Load the 68 facial landmark prediction model
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# ---------------- Video Input ----------------

# Load video file (use 0 for webcam)
cap = cv2.VideoCapture('01.mp4')

# Get frames per second of the video
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate delay between frames in milliseconds
delay = int(1000 / fps)

# ---------------- Main Loop ----------------
while True:

    # Read one frame from the video
    ret, frame = cap.read()

    # If no frame is returned, end the loop
    if not ret:
        break

    # Convert frame to grayscale (required by dlib)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ---------------- Detect Faces ----------------

    # Detect faces in the grayscale image
    faces = detector(gray)

    # Loop through all detected faces
    for face in faces:

        # Predict facial landmarks for the detected face
        shape = predictor(gray, face)

        # Convert landmark object to NumPy array (x, y)
        landmarks = face_utils.shape_to_np(shape)

        # ---------------- Draw Landmarks ----------------
        print(len(landmarks))

        # Loop through all 68 facial landmark points
        for (x, y) in landmarks:

            # Draw a small green circle on each landmark
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        # ---------------- Draw Face Bounding Box ----------------

        # Convert dlib rectangle to bounding box format
        x, y, w, h = face_utils.rect_to_bb(face)

        # Draw rectangle around the detected face
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )
    # Display the output frame
    cv2.imshow("Face Landmark Detection", frame)

    # Exit when ESC key is pressed
    if cv2.waitKey(delay) & 0xFF == 27:
        break

# Release video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
