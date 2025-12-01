import cv2
import mediapipe as mp
import joblib
import numpy as np
import time
from collections import deque, Counter

# Load gesture recognition model
model = joblib.load('models/gesture_model.pkl')

# Initialize MediaPipe Hands and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# Sentence and prediction tracking
sentence = ""
last_added = ""
prediction_buffer = deque(maxlen=10)
accepted_prediction = None
last_time = 0
delay_between_predictions = 2  # seconds

def get_landmark_vector(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if result.multi_hand_landmarks:
        landmarks = result.multi_hand_landmarks[0]
        vector = []
        for lm in landmarks.landmark:
            vector.extend([lm.x, lm.y, lm.z])
        return np.array(vector), True, result
    return None, False, None

cap = cv2.VideoCapture(0)
print("Starting gesture recognition. Press ESC to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    vector, hand_detected, results = get_landmark_vector(frame)
    prediction = None

    if hand_detected and vector is not None:
        prediction = model.predict([vector])[0]
        prediction_buffer.append(prediction)

        # Use smoothing: only accept if repeated enough
        most_common_pred, count = Counter(prediction_buffer).most_common(1)[0]

        if count > 7 and (most_common_pred != last_added or (time.time() - last_time) > delay_between_predictions):
            last_time = time.time()
            last_added = most_common_pred

            if most_common_pred.lower() == "space":
                sentence += " "
            else:
                sentence += most_common_pred

            prediction_buffer.clear()
            accepted_prediction = most_common_pred
    else:
        prediction_buffer.clear()

    # Draw hand landmarks
    if results and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    # Show prediction on webcam feed
    display_text = f"Prediction: {accepted_prediction if accepted_prediction else 'None'}"
    cv2.putText(frame, display_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Live Gesture Prediction", frame)

    # --- 📝 Chat-style sentence display ---
    max_line_length = 28  # characters per line
    lines = []
    current_line = ""

    for word in sentence.split():
        if len(current_line + word) <= max_line_length:
            current_line += word + " "
        else:
            lines.append(current_line.strip())
            current_line = word + " "
    if current_line:
        lines.append(current_line.strip())

    # Calculate image height based on number of lines
    line_height = 40
    padding = 20
    img_height = padding * 2 + line_height * len(lines)
    img_width = 800
    sentence_img = 255 * np.ones((img_height, img_width, 3), dtype=np.uint8)

    # Draw each line of sentence
    for i, line in enumerate(lines):
        y = padding + line_height * (i + 1) - 10
        cv2.putText(sentence_img, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow("Sentence", sentence_img)

    # Exit on ESC
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
