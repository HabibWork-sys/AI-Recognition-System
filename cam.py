from ultralytics import YOLO
import cv2
import torch
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading

# Check CUDA availability
print(f"CUDA Available: {torch.cuda.is_available()}")

# Load YOLO model
model = YOLO("yolov8s.pt")

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

class_names = model.names

# Use GPU if available
if torch.cuda.is_available():
    model.to("cuda")
    print("Using GPU acceleration!")
else:
    print("Using CPU")

# Initialize Tkinter GUI
root = tk.Tk()
root.title("YOLO Real-Time Detection")

# Variables for controlling the detection loop
is_running = False
demo_mode = False

# Function to start/stop detection
def toggle_detection():
    global is_running
    if is_running:
        is_running = False
        start_stop_button.config(text="Start Detection")
    else:
        is_running = True
        start_stop_button.config(text="Stop Detection")
        threading.Thread(target=detection_loop, daemon=True).start()

# Function to toggle demo mode
def toggle_demo_mode():
    global demo_mode
    demo_mode = not demo_mode
    demo_mode_button.config(text="Demo Mode: ON" if demo_mode else "Demo Mode: OFF")

# Function to run detection in a separate thread
def detection_loop():
    while is_running:
        ret, frame = cap.read()
        if not ret:
            break

        # Perform detection
        results = model.predict(
            frame,
            conf=0.5,
            imgsz=480,
            half=True,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        # Process results
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confidences = results[0].boxes.conf.cpu().numpy()
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            label = f"{class_names[cls_id]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display FPS
        fps = 1 / (results[0].speed['inference'] / 1000)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Convert frame to RGB for Tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.config(image=imgtk)

    if not is_running:
        video_label.config(image=None)

# Create UI elements
video_label = ttk.Label(root)
video_label.pack(padx=10, pady=10)

start_stop_button = ttk.Button(root, text="Start Detection", command=toggle_detection)
start_stop_button.pack(pady=5)

demo_mode_button = ttk.Button(root, text="Demo Mode: OFF", command=toggle_demo_mode)
demo_mode_button.pack(pady=5)

# Run the Tkinter main loop
root.mainloop()

# Release resources when the GUI is closed
cap.release()
cv2.destroyAllWindows()