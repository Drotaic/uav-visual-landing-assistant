# UAV Visual Landing Assistant

Vision-based landing system for UAVs using TensorFlow on Raspberry Pi, integrated with ArduPilot for autonomous landing control.

## 🎯 What It Does
- Detects landing zones in real-time using onboard camera
- Uses TensorFlow to infer safe areas based on visual input
- Commands Pixhawk via DroneKit to perform guided landings

## 💡 Stack
- TensorFlow (MobileNet or TFLite)
- OpenCV for image capture & preprocessing
- DroneKit + ArduPilot for flight control
- Raspberry Pi for onboard AI inference

## 📁 Structure
- `scripts/`: Vision + control logic
- `models/`: Trained TFLite model
- `notebooks/`: Model training workflow
- `assets/`: Example images or annotations

## 🔧 Requirements
- TensorFlow Lite
- OpenCV
- DroneKit
- ArduPilot (Pixhawk)
- Raspberry Pi
