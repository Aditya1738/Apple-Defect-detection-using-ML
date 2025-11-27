import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.layers import DepthwiseConv2D as OriginalDepthwiseConv2D
from tensorflow.keras.utils import get_custom_objects


class CustomDepthwiseConv2D(OriginalDepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')  
        super().__init__(*args, **kwargs)


get_custom_objects()['DepthwiseConv2D'] = CustomDepthwiseConv2D


fruit_model = tf.keras.models.load_model("F:/MTech Project/Main Project/best_fruit_model.h5")


apple_model = tf.keras.models.load_model("F:/MTech Project/Main Project/cnn_model_final_apple.keras")


FruitLabelDict = {0: "Apple", 1: "Apple", 2: "Apple"}


AppleLabelDict = {0: "Blotch_Apple", 1: "Normal_Apple", 2: "Rot_Apple", 3: "Scab_Apple"}


def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.expand_dims(img, axis=0)
    return img / 255.0  # Normalize


def detect_apple_regions(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = mask1 + mask2

    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(cnt)
            regions.append((x, y, w, h))
    return regions

# Start webcam
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Classify whole frame fruit type
    image_array = preprocess_frame(frame)
    fruit_probs = fruit_model.predict(image_array)[0]
    fruit_class = np.argmax(fruit_probs)
    fruit_confidence = fruit_probs[fruit_class]
    fruit_name = FruitLabelDict.get(fruit_class, "Unknown")

    if fruit_name == "Apple":
        apple_regions = detect_apple_regions(frame)
        for (x, y, w, h) in apple_regions:
            apple_crop = frame[y:y+h, x:x+w]
            apple_input = preprocess_frame(apple_crop)
            apple_probs = apple_model.predict(apple_input)[0]
            apple_class = np.argmax(apple_probs)
            apple_confidence = apple_probs[apple_class]
            apple_name = AppleLabelDict.get(apple_class, "Unknown")

            label_text = f"{apple_name} ({apple_confidence:.2f})"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        cv2.putText(frame, f"{fruit_name} ({fruit_confidence:.2f})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        label_text = f"{fruit_name} ({fruit_confidence:.2f})"
        cv2.putText(frame, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

    cv2.imshow("Live Fruit Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()