# 🦷 Tooth Detection using YOLOv11

This project implements an automated **tooth detection pipeline** on dental X-ray images using **YOLOv11**.  
The pipeline detects teeth, assigns **FDI numbering**, and organizes detections by quadrants with clear visual overlays.

---

## 📌 Project Overview
1. **Dataset Splitting**  
   - Divided dataset into **train:val:test = 8:1:1**.  

2. **Model Training**  
   - Trained a **YOLOv11 model** on annotated dental X-ray images.  
   - Used the **best-performing weights** (`best.pt`) for inference.  

3. **Inference Pipeline**  
   - Built a pipeline to process input X-ray images and detect bounding boxes + FDI tooth numbers.  

4. **Post-Processing**  
   - Added steps to:
     - Draw red dot markers for detected teeth.  
     - Label teeth with **FDI numbering system**.  
     - Organize detected teeth into quadrants: **UR, UL, LL, LR**.  

5. **Visualization & Output**  
   - Overlays quadrant-wise tooth lists on the image.  
   - Saves final processed image with detection results.

---

## 🚀 Installation
Clone the repository and install requirements:
```bash
git clone https://github.com/your-username/tooth-detection.git
cd tooth-detection
pip install -r requirements.txt
