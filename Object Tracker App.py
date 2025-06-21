#!/usr/bin/env python
# coding: utf-8

import cv2
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import depthai as dai
import numpy as np
import time
import datetime
from collections import defaultdict
import math
import os

# Model options
model_options = [
    "ai009yolo11n.vd.pt",
    "yolov8n.pt",
    "yolov8s.pt"
]

# Initialize YOLOv8 model with default selection
current_model = model_options[0]
#model = YOLO(current_model)

# Global variables
pipeline = None
device = None
running = False
tracking_id = None
conf_threshold = 0.4
track_history = defaultdict(lambda: [])
position_timestamps = defaultdict(lambda: [])
speed_history = defaultdict(lambda: [])
frame_count = 0
start_time = time.time()
fps = 0
previous_detected_ids = set()
log_level = "INFO"
total_objects = 0
show_path = False  # Default to disabled
auto_speed = False  # Default to disabled
video_file = None
cap = None
processing_mode = "camera"  # "camera" or "video"
speed_unit = "m/s"  # Default speed unit
recording = False
video_writer = None
output_video_path = None

# Fixed calibration settings
pixels_per_meter = 100
velocity_smoothing_window = 5

# Logger function
def log_message(level, category, message):
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    level_priority = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
    if level_priority[level] >= level_priority[log_level]:
        log_text.config(state=tk.NORMAL)
        log_text.insert(tk.END, f"[{timestamp}] ", "timestamp")
        log_text.insert(tk.END, f"[{level}] ", f"level_{level.lower()}")
        log_text.insert(tk.END, f"[{category}] ", f"category_{category.lower()}")
        log_text.insert(tk.END, f"{message}\n", "message")
        log_text.see(tk.END)
        log_text.config(state=tk.DISABLED)

def calculate_velocity(object_id):
    if len(track_history[object_id]) < 2 or len(position_timestamps[object_id]) < 2:
        return 0.0
    
    pos1 = track_history[object_id][-2]
    pos2 = track_history[object_id][-1]
    t1 = position_timestamps[object_id][-2]
    t2 = position_timestamps[object_id][-1]
    
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    displacement_pixels = math.sqrt(dx**2 + dy**2)
    displacement_meters = displacement_pixels / pixels_per_meter
    time_diff = t2 - t1
    
    if time_diff <= 0:
        return 0.0
    
    velocity = displacement_meters / time_diff
    speed_history[object_id].append(velocity)
    
    if len(speed_history[object_id]) > velocity_smoothing_window:
        speed_history[object_id].pop(0)
    
    # Convert units if needed
    speed_value = sum(speed_history[object_id]) / len(speed_history[object_id])
    if speed_unit == "km/h":
        speed_value = speed_value * 3.6  # m/s to km/h
    elif speed_unit == "mph":
        speed_value = speed_value * 2.237  # m/s to mph
    
    return speed_value

def select_video_file():
    global video_file, processing_mode, cap
    file_path = filedialog.askopenfilename(
        filetypes=[("Video Files", ".mp4 *.avi *.mov *.mkv"), ("All Files", ".*")]
    )
    if file_path:
        # Test if video can be opened
        test_cap = cv2.VideoCapture(file_path)
        if not test_cap.isOpened():
            messagebox.showerror("Error", "Could not open video file")
            test_cap.release()
            return False
            
        # Get video properties
        width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = test_cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count/fps if fps > 0 else 0
        
        log_message("INFO", "VIDEO", f"Resolution: {width}x{height}")
        log_message("INFO", "VIDEO", f"FPS: {fps:.2f}")
        log_message("INFO", "VIDEO", f"Duration: {duration:.2f} seconds")
        
        test_cap.release()
        
        video_file = file_path
        processing_mode = "video"
        cap = cv2.VideoCapture(video_file)
        log_message("INFO", "SYSTEM", f"Video file selected: {os.path.basename(video_file)}")
        status_label.config(text=f"Video: {os.path.basename(video_file)}", foreground="blue")
        return True
    return False

def switch_to_camera():
    global processing_mode, cap, video_file
    processing_mode = "camera"
    if cap is not None:
        cap.release()
        cap = None
    video_file = None
    log_message("INFO", "SYSTEM", "Switched to camera mode")
    status_label.config(text="Ready for camera stream", foreground="blue")

def update_tracking_id():
    global tracking_id, speed_history, results  # Add results to global declarations
    
    try:
        new_id = int(entry_id.get())
        if tracking_id != new_id:
            old_id = tracking_id
            tracking_id = new_id
            status_label.config(text=f"Tracking ID: {tracking_id}", foreground="green")
            
            # Log the class of the tracked object if it exists
            object_class = None
            
            # Check if results exists before trying to use it
            if 'results' in globals() and results is not None:
                for result in results:
                    if result.boxes.id is not None:
                        ids = result.boxes.id.cpu().numpy().astype(int)
                        if tracking_id in ids:
                            idx = list(ids).index(tracking_id)
                            cls = result.boxes.cls.cpu().numpy().astype(int)[idx]
                            object_class = model.names[cls]
                            break
                            
                if object_class:
                    log_message("INFO", "TRACKING", f"Now tracking ID {tracking_id}, Class: {object_class}")
                else:
                    log_message("INFO", "TRACKING", f"Tracking target changed from ID {old_id if old_id is not None else 'None'} to ID {tracking_id}")
            else:
                log_message("INFO", "TRACKING", f"Tracking target changed from ID {old_id if old_id is not None else 'None'} to ID {tracking_id}")
                
            speed_label.config(text="N/A")
            if tracking_id in speed_history:
                speed_history[tracking_id] = []
    except ValueError:
        tracking_id = None
        status_label.config(text="Invalid ID", foreground="red")
        log_message("WARNING", "TRACKING", "Invalid tracking ID entered")
        speed_label.config(text="N/A")
        
        
def update_conf_threshold(val):
    global conf_threshold
    new_threshold = float(val)
    if new_threshold != conf_threshold:
        old_threshold = conf_threshold
        conf_threshold = new_threshold
        conf_label.config(text=f"{conf_threshold:.1f}")
        log_message("DEBUG", "DETECTION", f"Confidence threshold changed from {old_threshold:.1f} to {conf_threshold:.1f}")

def start_stream():
    global pipeline, device, running, frame_count, start_time, model

    if running:
        stop_stream()

    # Load the selected model freshly each time
    model = YOLO(current_model)

    if processing_mode == "camera":
        pipeline = dai.Pipeline()
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.setPreviewSize(800, 600)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        cam_rgb.preview.link(xout_rgb.input)

        try:
            device = dai.Device(pipeline)
            running = True
            frame_count = 0
            start_time = time.time()
            status_label.config(text=f"Running with {current_model}", foreground="green")
            log_message("INFO", "SYSTEM", f"Camera stream started with model {current_model}")
            update_video()
        except Exception as e:
            status_label.config(text=f"Error: {str(e)[:30]}...", foreground="red")
            log_message("ERROR", "SYSTEM", f"Failed to start camera stream: {str(e)}")

    elif processing_mode == "video" and cap is not None:
        running = True
        frame_count = 0
        start_time = time.time()
        status_label.config(text=f"Processing video with {current_model}", foreground="green")
        log_message("INFO", "SYSTEM", f"Video processing started with model {current_model}")
        update_video()


def stop_stream():
    global running, device, cap, recording, video_writer
    if running:
        running = False
        if device:
            device.close()
            device = None
        if cap:
            cap.release()
            cap = None
        # Stop recording if active
        if recording and video_writer is not None:
            video_writer.release()
            video_writer = None
            recording = False
            record_btn.config(text="Record Video")
            log_message("INFO", "SYSTEM", f"Video recording saved to {output_video_path}")
        
        # Update camera status to Offline
        camera_status_label.config(text="Offline", foreground="red")
        
        status_label.config(text="Stream stopped", foreground="orange")
        log_message("INFO", "SYSTEM", "Stream stopped")
        speed_label.config(text="N/A")
        
        # Clear the canvas when stream is stopped
        canvas.delete("all")
        canvas.create_text(canvas.winfo_width()//2, canvas.winfo_height()//2, 
                         text="Stream stopped", fill="white", font=("Helvetica", 14))
def reset_tracking():
    global track_history, tracking_id, previous_detected_ids, position_timestamps, speed_history, total_objects
    track_history.clear()
    position_timestamps.clear()
    speed_history.clear()
    previous_detected_ids = set()
    old_id = tracking_id
    tracking_id = None
    total_objects = 0
    entry_id.delete(0, tk.END)
    status_label.config(text="Tracking reset", foreground="orange")
    speed_label.config(text="N/A")
    log_message("INFO", "TRACKING", f"Tracking reset. Previous target was ID {old_id if old_id is not None else 'None'}")

def take_snapshot():
    if hasattr(canvas, 'imgtk'):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{timestamp}.jpg"
        try:
            img = ImageTk.getimage(canvas.imgtk)
            if img.mode == "RGBA":
                img = img.convert("RGB")
            img.save(filename)
            log_message("INFO", "SYSTEM", f"Snapshot saved as {filename}")
        except Exception as e:
            log_message("ERROR", "SYSTEM", f"Failed to save snapshot: {str(e)}")
    else:
        log_message("WARNING", "SYSTEM", "Cannot take snapshot: No frame available")

def change_speed_unit(event):
    global speed_unit
    speed_unit = speed_unit_var.get()
    log_message("INFO", "SYSTEM", f"Speed unit changed to {speed_unit}")
    # Update speed display if tracking active
    if tracking_id is not None and tracking_id in speed_history and len(speed_history[tracking_id]) > 0:
        velocity = calculate_velocity(tracking_id)
        speed_label.config(text=f"{velocity:.2f} {speed_unit}")

def update_video():
    global running, frame_count, start_time, fps, total_objects, previous_detected_ids
    
    if not running:
        return
    
    frame = None
    
    if processing_mode == "camera" and device:
        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        in_rgb = q_rgb.tryGet()
        if in_rgb is not None:
            frame = in_rgb.getCvFrame()
            # Update camera status to Online
            camera_status_label.config(text="Online", foreground="green")
    elif processing_mode == "video" and cap is not None:
        ret, frame = cap.read()
        if not ret:
            log_message("INFO", "SYSTEM", "Video processing completed")
            stop_stream()
            return
        # Update camera status to Online (for video playback)
        camera_status_label.config(text="Online", foreground="green")
    
    if frame is not None:
        try:
            # Get current canvas dimensions each frame to ensure proper sizing
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            
            # Process frame
            current_time = time.time()
            annotated_frame = frame.copy()
            
            results = model.track(
                frame,
                persist=True,
                conf=conf_threshold,
                tracker="bytetrack.yaml",
                verbose=False
            )

            frame_count += 1
            elapsed = current_time - start_time
            if elapsed > 1:
                fps = frame_count / elapsed
                fps_label.config(text=f"FPS: {fps:.1f}")
                log_message("DEBUG", "SYSTEM", f"Current FPS: {fps:.1f}")
                frame_count = 0
                start_time = current_time

            if results and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                classes = results[0].boxes.cls.cpu().numpy().astype(int)
                confs = results[0].boxes.conf.cpu().numpy()

                current_ids = set(ids)
                new_ids = current_ids - previous_detected_ids
                
                # Log newly detected objects
                for obj_id in new_ids:
                    idx = list(ids).index(obj_id)
                    cls = classes[idx]
                    class_name = model.names[cls]
                    log_message("INFO", "DETECTION", f"Object detected: ID {obj_id}, Class: {class_name}")
                
                total_objects += len(new_ids)
                
                for box, obj_id, cls, conf in zip(boxes, ids, classes, confs):
                    x1, y1, x2, y2 = box
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    track_history[obj_id].append(center)
                    position_timestamps[obj_id].append(current_time)

                    if len(track_history[obj_id]) > 50:
                        track_history[obj_id].pop(0)
                        position_timestamps[obj_id].pop(0)

                    if obj_id == tracking_id and auto_speed:
                        velocity = calculate_velocity(obj_id)
                        speed_label.config(text=f"{velocity:.2f} {speed_unit}")

                        # Log the tracking information periodically (every 50 frames)
                        if frame_count % 50 == 0:
                            class_name = model.names[cls]
                            log_message("DEBUG", "TRACKING", f"Currently tracking: ID {obj_id}, Class: {class_name}, Speed: {velocity:.2f} {speed_unit}")

                    # Modified path drawing logic
                    if show_path:
                        # Only draw path for specific ID if tracking_id is set
                        if tracking_id is not None:
                            if obj_id == tracking_id:
                                for i in range(1, len(track_history[obj_id])):
                                    cv2.line(annotated_frame,
                                            track_history[obj_id][i-1],
                                            track_history[obj_id][i],
                                            (0, 255, 255), 2)
                        else:
                            # Draw for all objects if no specific ID is tracked
                            for i in range(1, len(track_history[obj_id])):
                                cv2.line(annotated_frame,
                                        track_history[obj_id][i-1],
                                        track_history[obj_id][i],
                                        (0, 255, 255), 2)

                    color = (0, 255, 0)
                    thickness = 1
                    if obj_id == tracking_id:
                        color = (0, 0, 255)
                        thickness = 2

                    # Draw rectangle with same thickness as before
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)

                    # Draw label with increased font weight for boldness
                    label = f"ID: {obj_id} {model.names[cls]} {conf:.2f}"
                    # Use thicker text (2 instead of 1) while keeping font scale the same
                    cv2.putText(annotated_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                

                previous_detected_ids = current_ids.copy()

            # Record frame if recording is active
            if recording and video_writer is not None:
                video_writer.write(annotated_frame)

            # Convert frame for display
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Make the stream fit exactly to the canvas size
            if canvas_width > 0 and canvas_height > 0:  # Prevent resizing to zero dimensions
                img = img.resize((canvas_width, canvas_height), Image.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)
                
                # Update display - fit the image to the canvas
                canvas.delete("all")
                canvas.imgtk = imgtk
                canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            
        except Exception as e:
            error_msg = f"Processing error: {str(e)}"
            log_message("ERROR", "SYSTEM", error_msg)
            canvas.delete("all")
            canvas.create_text(canvas.winfo_width()//2, canvas.winfo_height()//2, 
                             text=error_msg, fill="red", anchor=tk.CENTER, font=("Helvetica", 12))

    if running:
        root.after(10, update_video)

def change_model(event):
    global model, current_model
    selected_model = model_var.get()
    if selected_model != current_model:
        try:
            old_model = current_model
            current_model = selected_model
            model = YOLO(current_model)
            log_message("INFO", "MODEL", f"Model changed from {old_model} to {current_model}")
            status_label.config(text=f"Model: {current_model}", foreground="blue")
        except Exception as e:
            log_message("ERROR", "MODEL", f"Failed to load model {selected_model}: {str(e)}")
            current_model = old_model
            model_var.set(current_model)

def change_log_level(event):
    global log_level
    log_level = log_level_var.get()
    log_message("INFO", "SYSTEM", f"Log level changed to {log_level}")

def toggle_auto_speed():
    global auto_speed
    auto_speed = auto_speed_var.get()
    log_message("INFO", "SYSTEM", f"Auto speed update {'enabled' if auto_speed else 'disabled'}")

def toggle_show_path():
    global show_path
    show_path = show_path_var.get()
    log_message("INFO", "SYSTEM", f"Path visualization {'enabled' if show_path else 'disabled'}")

def toggle_recording():
    global recording, video_writer, output_video_path
    
    if recording:
        # Stop recording
        if video_writer is not None:
            video_writer.release()
            video_writer = None
            log_message("INFO", "SYSTEM", f"Video recording saved to {output_video_path}")
            record_btn.config(text="Record Video")
        recording = False
    else:
        # Start recording
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_video_path = f"recording_{timestamp}.mp4"
        
        # Determine frame size based on current mode
        frame_size = (800, 600)  # Default
        if processing_mode == "video" and cap is not None:
            frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 
                         int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        
        # Create VideoWriter
        video_writer = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            30,  # FPS
            frame_size
        )
        
        recording = True
        record_btn.config(text="Stop Recording")
        log_message("INFO", "SYSTEM", "Video recording started")

def export_logs():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"logs_{timestamp}.txt"
    try:
        with open(filename, 'w') as f:
            f.write(log_text.get(1.0, tk.END))
        log_message("INFO", "SYSTEM", f"Logs exported to {filename}")
    except Exception as e:
        log_message("ERROR", "SYSTEM", f"Failed to export logs: {str(e)}")
        
        
def update_datetime_display():
    current_datetime = datetime.datetime.now().strftime("%A, %B %d, %Y - %H:%M:%S")
    datetime_label.config(text=current_datetime)
    root.after(1000, update_datetime_display)  # Update every second

        

def show_about():
    about_window = tk.Toplevel(root)
    about_window.title("About Object Detection and Speed Tracker")
    about_window.geometry("400x300")
    about_window.resizable(False, False)
    about_window.transient(root)
    about_window.grab_set()
    
    # Create frame with scrollbar
    about_frame = ttk.Frame(about_window)
    about_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    # Add scrollbar
    about_scroll = ttk.Scrollbar(about_frame, orient="vertical")
    about_scroll.pack(side=tk.RIGHT, fill=tk.Y)
    
    # Text widget with scrollbar
    about_text = tk.Text(about_frame, wrap=tk.WORD, yscrollcommand=about_scroll.set)
    about_text.pack(fill=tk.BOTH, expand=True)
    about_scroll.config(command=about_text.yview)
    
    # About content
    about_content = """Object Detection and Speed Tracker
Version 1.0.0

This application uses computer vision models to detect objects 
in video streams and track their movement and measure speed.

Features:
- Multiple detection models
- Real-time speed calculation
- Object tracking
- Video and camera input support
- Snapshot capability

Developed by: Saifullah, Shahmeer

© 2025 All Rights Reserved
"""
    
    about_text.insert(tk.END, about_content)
    about_text.config(state=tk.DISABLED)
    
    # OK button
    ttk.Button(about_window, text="OK", command=about_window.destroy).pack(pady=10)

def on_closing():
    stop_stream()
    root.destroy()

def on_canvas_configure(event):
    # Force redraw of the canvas with new dimensions
    if running and hasattr(canvas, 'imgtk'):
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        canvas.delete("all")
        canvas.create_image(canvas_width//2, canvas_height//2, anchor=tk.CENTER, image=canvas.imgtk)
        
        
        
# Main GUI setup
root = tk.Tk()
root.title("Object Detection and Speed Tracker")
root.geometry("1000x800")
root.protocol("WM_DELETE_WINDOW", on_closing)

# Apply custom styles with LARGER font size
style = ttk.Style()
font_size = 11  # Increased font size from 10 to 11
style.configure('TButton', font=('Helvetica', font_size), padding=(12, 5))
style.configure('TLabel', font=('Helvetica', font_size))
style.configure('TCheckbutton', font=('Helvetica', font_size))
style.configure('TCombobox', font=('Helvetica', font_size))
style.configure('TEntry', font=('Helvetica', font_size))
style.configure('TLabelframe.Label', font=('Helvetica', font_size, 'bold'))
style.configure('TNotebook.Tab', font=('Helvetica', font_size))

# Configure grid for main window
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=0)  # Header
root.grid_rowconfigure(1, weight=1)  # Body

# Top bar with title and about button
top_bar = ttk.Frame(root, padding="5")
top_bar.grid(row=0, column=0, sticky="ew")
top_bar.grid_columnconfigure(0, weight=1)
top_bar.grid_columnconfigure(1, weight=1)
top_bar.grid_columnconfigure(2, weight=1)

top_bar.grid_columnconfigure(2, weight=1)  # Add column for datetime


# About button on LEFT side
about_btn = ttk.Button(top_bar, text="About", command=show_about, width=8)
about_btn.grid(row=0, column=0, sticky="w", padx=5, pady=5)

title_label = ttk.Label(top_bar, text="Real Time Object Tracking and Video Surveillance", font=("Helvetica", 16, "bold"))
title_label.grid(row=0, column=1, sticky="n", pady=10)

# Add datetime display on the right side
datetime_label = ttk.Label(top_bar, text="", font=("Helvetica", 10))
datetime_label.grid(row=0, column=2, sticky="e", padx=10)

# Main body - COMPLETELY FIXED LAYOUT (NO SASHES)
main_frame = ttk.Frame(root)
main_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
main_frame.grid_columnconfigure(0, weight=95)  # Video panel 75%
main_frame.grid_columnconfigure(1, weight=5)  # Settings panel 25%
main_frame.grid_rowconfigure(0, weight=70)     # Upper section 70%
main_frame.grid_rowconfigure(1, weight=30)     # Logs section 30%

# Video panel (left side, top) - fixed at 75% width, 70% height
video_frame = ttk.Frame(main_frame, relief="groove", borderwidth=1)
video_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=(0, 5))

# Video canvas WITHOUT any scrollbar below
canvas = tk.Canvas(video_frame, bg="black")
canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
canvas.bind("<Configure>", on_canvas_configure)

# Settings panel (right side, top) - fixed at 25% width, 70% height
settings_panel = ttk.Frame(main_frame, relief="groove", borderwidth=1)
settings_panel.grid(row=0, column=1, sticky="nsew", pady=(0, 5))

# Settings title
settings_title = ttk.Label(settings_panel, text="Settings", font=("Helvetica", font_size+2, "bold"))
settings_title.pack(fill=tk.X, padx=5, pady=10)

# Configure right panel with notebook for organized settings
settings_notebook = ttk.Notebook(settings_panel)
settings_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

# Model settings tab
model_tab = ttk.Frame(settings_notebook, padding=10)
settings_notebook.add(model_tab, text="Model")

# Add scrollbars to model tab
model_canvas = tk.Canvas(model_tab)
model_scrollbar = ttk.Scrollbar(model_tab, orient="vertical", command=model_canvas.yview)
model_scrollable_frame = ttk.Frame(model_canvas)

model_scrollable_frame.bind(
    "<Configure>",
    lambda e: model_canvas.configure(scrollregion=model_canvas.bbox("all"))
)

model_canvas.create_window((0, 0), window=model_scrollable_frame, anchor="nw")
model_canvas.configure(yscrollcommand=model_scrollbar.set)

model_canvas.pack(side="left", fill="both", expand=True)
model_scrollbar.pack(side="right", fill="y")

# INCREASED Grid padding for model tab
grid_pady = 8  # Increased spacing between rows from 4 to 8
grid_padx = 5  # Slightly increased horizontal padding

# Model selection
ttk.Label(model_scrollable_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, pady=grid_pady)
model_var = tk.StringVar(value=current_model)
model_dropdown = ttk.Combobox(model_scrollable_frame, textvariable=model_var, values=model_options, width=15, state="readonly")
model_dropdown.grid(row=0, column=1, sticky="ew", pady=grid_pady, padx=grid_padx)
model_dropdown.bind("<<ComboboxSelected>>", change_model)

# Log level
ttk.Label(model_scrollable_frame, text="Log Level:").grid(row=1, column=0, sticky=tk.W, pady=grid_pady)
log_level_var = tk.StringVar(value=log_level)
log_level_dropdown = ttk.Combobox(model_scrollable_frame, textvariable=log_level_var, 
                                 values=["DEBUG", "INFO", "WARNING", "ERROR"], 
                                 width=15, state="readonly")
log_level_dropdown.grid(row=1, column=1, sticky="ew", pady=grid_pady, padx=grid_padx)
log_level_dropdown.bind("<<ComboboxSelected>>", change_log_level)

# Confidence threshold
ttk.Label(model_scrollable_frame, text="Confidence:").grid(row=2, column=0, sticky=tk.W, pady=grid_pady)
conf_frame = ttk.Frame(model_scrollable_frame)
conf_frame.grid(row=2, column=1, sticky="ew", pady=grid_pady, padx=grid_padx)
conf_slider = ttk.Scale(conf_frame, from_=0.1, to=0.9, value=conf_threshold)
conf_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0,5))
conf_label = ttk.Label(conf_frame, text=f"{conf_threshold:.1f}", width=4)
conf_label.pack(side=tk.RIGHT)
conf_slider.config(command=update_conf_threshold)

# Source buttons in a single row - SMALLER buttons
source_frame = ttk.Frame(model_scrollable_frame)
source_frame.grid(row=3, column=0, columnspan=2, sticky="ew", pady=grid_pady)
ttk.Button(source_frame, text="Use Camera", command=switch_to_camera, width=10).pack(side=tk.LEFT, padx=(0, 5))
ttk.Button(source_frame, text="Load Video", command=select_video_file, width=10).pack(side=tk.RIGHT)

# Tracking tab
tracking_tab = ttk.Frame(settings_notebook, padding=10)
settings_notebook.add(tracking_tab, text="Tracking")

# Add scrollbars to tracking tab
tracking_canvas = tk.Canvas(tracking_tab)
tracking_scrollbar = ttk.Scrollbar(tracking_tab, orient="vertical", command=tracking_canvas.yview)
tracking_scrollable_frame = ttk.Frame(tracking_canvas)

tracking_scrollable_frame.bind(
    "<Configure>",
    lambda e: tracking_canvas.configure(scrollregion=tracking_canvas.bbox("all"))
)

tracking_canvas.create_window((0, 0), window=tracking_scrollable_frame, anchor="nw")
tracking_canvas.configure(yscrollcommand=tracking_scrollbar.set)

tracking_canvas.pack(side="left", fill="both", expand=True)
tracking_scrollbar.pack(side="right", fill="y")

# CAMERA STATUS at the top of tracking tab
camera_status_frame = ttk.Frame(tracking_scrollable_frame)
camera_status_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=grid_pady)
ttk.Label(camera_status_frame, text="Camera Status:", font=("Helvetica", font_size)).pack(side=tk.LEFT, pady=5)
camera_status_label = ttk.Label(camera_status_frame, text="Offline", foreground="red", font=("Helvetica", font_size, "bold"))
camera_status_label.pack(side=tk.RIGHT, pady=5)

# Speed unit
ttk.Label(tracking_scrollable_frame, text="Speed Unit:").grid(row=1, column=0, sticky=tk.W, pady=grid_pady)
speed_unit_var = tk.StringVar(value=speed_unit)
speed_unit_dropdown = ttk.Combobox(tracking_scrollable_frame, textvariable=speed_unit_var, 
                                  values=["m/s", "km/h", "mph"], 
                                  width=15, state="readonly")
speed_unit_dropdown.grid(row=1, column=1, sticky="ew", pady=grid_pady, padx=grid_padx)
speed_unit_dropdown.bind("<<ComboboxSelected>>", change_speed_unit)

# Toggle options with enabled state variables
auto_speed_var = tk.BooleanVar(value=False)
show_path_var = tk.BooleanVar(value=False)

def toggle_auto_speed():
    global auto_speed
    auto_speed = not auto_speed
    log_message("INFO", "SYSTEM", f"Auto speed update {'enabled' if auto_speed else 'disabled'}")

def toggle_show_path():
    global show_path
    show_path = not show_path
    log_message("INFO", "SYSTEM", f"Path visualization {'enabled' if show_path else 'disabled'}")


# Toggle options with checkboxes tied to variables
ttk.Checkbutton(tracking_scrollable_frame, text="Auto Speed Update", variable=auto_speed_var, 
               command=toggle_auto_speed).grid(row=2, column=0, columnspan=2, sticky="w", pady=grid_pady)
ttk.Checkbutton(tracking_scrollable_frame, text="Show Path", variable=show_path_var,
               command=toggle_show_path).grid(row=3, column=0, columnspan=2, sticky="w", pady=grid_pady)

# NOW: Tracking ID ABOVE the speed display
ttk.Label(tracking_scrollable_frame, text="Track ID:").grid(row=5, column=0, sticky=tk.W, pady=grid_pady)
track_frame = ttk.Frame(tracking_scrollable_frame)
track_frame.grid(row=5, column=1, sticky="ew", pady=grid_pady, padx=grid_padx)
entry_id = ttk.Entry(track_frame, width=15)
entry_id.pack(side=tk.LEFT, fill=tk.X, expand=True)
entry_id.insert(0, "1")
ttk.Button(track_frame, text="Set", command=update_tracking_id, width=6).pack(side=tk.RIGHT, padx=(5, 0))

# Bind Enter key to Set button functionality
entry_id.bind("<Return>", lambda event: update_tracking_id())

# Status display - AFTER tracking ID (now at row 6)
status_frame = ttk.Frame(tracking_scrollable_frame)
status_frame.grid(row=6, column=0, columnspan=2, sticky="ew", pady=grid_pady)
ttk.Label(status_frame, text="Speed of Tracked Object:", font=("Helvetica", font_size)).pack(side=tk.LEFT, pady=5)
speed_label = ttk.Label(status_frame, text="N/A", foreground="darkgreen", font=("Helvetica", font_size, "bold"))
speed_label.pack(side=tk.RIGHT, pady=5)

# Action buttons in 2 rows, with SMALLER buttons
action_frame = ttk.Frame(tracking_scrollable_frame)
action_frame.grid(row=7, column=0, columnspan=2, sticky="ew", pady=grid_pady)
action_frame.grid_columnconfigure(0, weight=1)
action_frame.grid_columnconfigure(1, weight=1)

# First row of buttons - smaller size
ttk.Button(action_frame, text="Start", command=start_stream, width=8).grid(row=0, column=0, padx=5, pady=5, sticky="ew")
ttk.Button(action_frame, text="Stop", command=stop_stream, width=8).grid(row=0, column=1, padx=5, pady=5, sticky="ew")

# Second row of buttons - smaller size
ttk.Button(action_frame, text="Reset", command=reset_tracking, width=8).grid(row=1, column=0, padx=5, pady=5, sticky="ew")
ttk.Button(action_frame, text="Snapshot", command=take_snapshot, width=8).grid(row=1, column=1, padx=5, pady=5, sticky="ew")

record_btn = ttk.Button(action_frame, text="Record Video", command=toggle_recording, width=8)
record_btn.grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky="ew")

# Logs panel (bottom) - fixed at 30% height, spans full width
logs_frame = ttk.Frame(main_frame, relief="groove", borderwidth=1)
logs_frame.grid(row=1, column=0, columnspan=2, sticky="nsew")

# Log text area with MINIMAL padding
log_text_frame = ttk.Frame(logs_frame)
log_text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

log_text_v_scroll = ttk.Scrollbar(log_text_frame, orient="vertical")
log_text = scrolledtext.ScrolledText(
    log_text_frame, 
    wrap=tk.WORD,
    yscrollcommand=log_text_v_scroll.set,
    height=5  # Smaller height
)

log_text_v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
log_text.pack(fill=tk.BOTH, expand=True)
log_text_v_scroll.config(command=log_text.yview)
log_text.config(state=tk.DISABLED)

# Configure text tags for log styling
log_text.tag_configure("timestamp", foreground="darkgray")
log_text.tag_configure("level_debug", foreground="gray")
log_text.tag_configure("level_info", foreground="black")
log_text.tag_configure("level_warning", foreground="orange")
log_text.tag_configure("level_error", foreground="red", font=("TkDefaultFont", font_size, "bold"))
log_text.tag_configure("category_model", foreground="blue")
log_text.tag_configure("category_detection", foreground="green")
log_text.tag_configure("category_tracking", foreground="purple")
log_text.tag_configure("category_system", foreground="brown")
log_text.tag_configure("category_velocity", foreground="darkgreen")
log_text.tag_configure("message", foreground="black")

# Status bar at the bottom
status_bar = ttk.Frame(root)
status_bar.grid(row=2, column=0, sticky="ew")
status_bar.grid_columnconfigure(0, weight=1)  # Status label
status_bar.grid_columnconfigure(1, weight=1)  # FPS
status_bar.grid_columnconfigure(2, weight=1)  # Clear logs button

# Status label on left
status_label = ttk.Label(status_bar, text="Ready", foreground="blue")
status_label.grid(row=0, column=0, sticky="w", padx=10)

# FPS CENTERED
fps_label = ttk.Label(status_bar, text="FPS: 0.0")
fps_label.grid(row=0, column=1, sticky="n", padx=10)

# Clear Logs button to bottom right in status bar
clear_logs_btn = ttk.Button(status_bar, text="Clear Logs", width=10,
                           command=lambda: log_text.config(state=tk.NORMAL) or log_text.delete(1.0, tk.END) or log_text.config(state=tk.DISABLED))
clear_logs_btn.grid(row=0, column=2, sticky="e", padx=10)

export_logs_btn = ttk.Button(status_bar, text="Export Logs", width=10, command=export_logs)
export_logs_btn.grid(row=0, column=3, sticky="e", padx=10)

# Initialize with welcome log message
log_message("INFO", "SYSTEM", "Application launched.")
log_message("INFO", "MODEL", f"Loaded initial model: {current_model}")
log_message("INFO", "VELOCITY", "Speed calculation ready (100 pixels = 1 meter)")

update_datetime_display()  # Start the datetime update cycle

root.mainloop()