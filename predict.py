import cv2
import mediapipe as mp
import numpy as np
import csv

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
cap = cv2.VideoCapture(0)

# Ask user for letter input
letter = input("Enter the letter you want to record (A-Z): ").upper()
if len(letter) != 1 or not letter.isalpha():
    print("Please enter a single letter A-Z")
    exit()

data = []

print(f"Starting data collection for letter '{letter}'. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            data.append([letter] + landmarks)

    cv2.putText(frame, f"Recording: {letter}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Collecting", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Save collected data to CSV file named like data_A.csv
filename = f"data_{letter}.csv"
with open(filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['label'] + [f'{i}' for i in range(63)])
    writer.writerows(data)

print(f"Data for letter '{letter}' saved to {filename}")
