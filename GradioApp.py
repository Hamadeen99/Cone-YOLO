import gradio as gr
from ultralytics import YOLO
import cv2
import os
import uuid
import time
import numpy as np
import plotly.graph_objects as go


model = YOLO("Cone_YOLO11n.pt")  

def detect_cones(video, progress=gr.Progress()):

    os.makedirs("output", exist_ok=True)
    output_vid = f"output/{uuid.uuid4().hex}.mp4"

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    frame_counter = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    in_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer_vid = cv2.VideoWriter(output_vid, cv2.VideoWriter_fourcc(*'mp4v'), fps, (in_width, in_height))

    total_cones = 0
    snapshots = []
    fps_values = []

    start_time = time.time()

    for i in progress.tqdm(range(frame_counter)):
        ret, frame = cap.read()
        if not ret:
            break

        fstart = time.time()
        results = model.predict(frame, conf=0.5, save=False)
        fstop = time.time()
        fps_values.append(1 / (fstop - fstart))

        boxes = results[0].boxes
        cone_count = sum(1 for box in boxes if float(box.conf[0]) >= 0.8)
        total_cones += cone_count

        annotated = results[0].plot()
        writer_vid.write(annotated)

        
        if i % max(1, frame_counter // 5) == 0:
            snapshots.append(annotated.copy())

    cap.release()
    writer_vid.release()
    duration_time = time.time() - start_time
    avg_fps = frame_counter / duration_time 

    summary_info = f"""
### 
 **Input Video Information**
- Resolution: {in_width}x{in_height}  
- Frames: {frame_counter}  
- Original FPS: {fps:.1f}  
- Processing FPS: {avg_fps:.2f}  

 **Detection Summary**
- Total traffic cones detected: **{total_cones}**  
- Runtime: {duration_time:.2f} seconds
"""

    return output_vid, summary_info, "Cone_YOLO11n.pt", snapshots



my_theme = gr.themes.Soft(
    primary_hue="orange",
    font=["Inter", "sans-serif"],
    spacing_size=gr.themes.sizes.spacing_md,
    radius_size=gr.themes.sizes.radius_lg
)


with gr.Blocks(theme=my_theme, title="Traffic Cone Detection") as demo:
    gr.Markdown("""
    <h1 align="center"> Traffic Cone Detection with YOLO11 Nano  </h1>
    <p align="center">Upload a video to detect traffic cones using a fine-tuned YOLO11n model.</p>  

    -  Download Annotated video
    -  Input Video Information  
    -  Detection summary  
    -  Snapshots fromyour output video  
    -  Download the fine-tuned model to use it
    """)

    video_input = gr.Video(label=" Upload Video", interactive=True)
    detect_b = gr.Button(" Start")

    output_video = gr.File(label=" Download Output Video")
    output_summary = gr.Markdown()
    model_file = gr.File(label=" Download (Cone_YOLO11n.pt)")
    screenshot = gr.Gallery(label=" Snapshots", columns=4, height=220)

    detect_b.click(
        fn=detect_cones,
        inputs=[video_input],
        outputs=[output_video, output_summary, model_file, screenshot]
    )

    gr.Markdown("""
---
<h1 align="center">Demo APP to test the Fine tuned model on traffic cones (Ideal for Self-driving robots & cars) </h1>
""")

demo.launch()
