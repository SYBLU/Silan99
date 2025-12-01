
import cv2
import os

gesture_name = input("Enter the gesture label (e.g., hello): ")
save_dir = f"dataset/{gesture_name}"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0
print("Press SPACE to capture images. ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture - " + gesture_name, frame)
    key = cv2.waitKey(1)

    if key == 27:  # ESC key
        break
    elif key == 32:  # SPACE key
        count += 1
        file_path = os.path.join(save_dir, f"{gesture_name}_{count}.jpg")
        cv2.imwrite(file_path, frame)
        print(f"Saved: {file_path}")

cap.release()
cv2.destroyAllWindows()
