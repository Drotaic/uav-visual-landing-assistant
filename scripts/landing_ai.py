import cv2
import numpy as np
import tensorflow as tf
from dronekit import connect, VehicleMode, LocationGlobalRelative
import time

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="models/landing_zone_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Connect to vehicle
vehicle = connect('127.0.0.1:14550', wait_ready=True)

def capture_frame():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None

def detect_landing_zone(frame):
    resized = cv2.resize(frame, (224, 224))
    input_data = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0][0] > 0.5  # assume binary classification: 1 = safe, 0 = unsafe

def land_drone():
    print("Landing...")
    vehicle.mode = VehicleMode("LAND")

def main_loop():
    print("Running visual landing assistant...")
    while True:
        frame = capture_frame()
        if frame is not None and detect_landing_zone(frame):
            print("Landing zone detected.")
            land_drone()
            break
        else:
            print("No safe zone detected.")
        time.sleep(1)

if __name__ == "__main__":
    main_loop()
    vehicle.close()
