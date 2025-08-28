import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
model = YOLO("best.pt")

def draw_text_with_outline(img, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX,
                           font_scale=0.8, text_color=(255,255,255),
                           outline_color=(0,0,0), thickness=2):
    """ Draw outlined text for visibility on X-ray """
    cv2.putText(img, text, pos, font, font_scale, outline_color, thickness+2, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, font_scale, text_color, thickness, cv2.LINE_AA)


def process_image(img_path, save_path="output.jpg"):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    results = model(img_path)

    detections = []
    for r in results:
        boxes = r.boxes.xywh.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy()

        names = [
    '13','23','33','43','21','41','31','11',
    '16','26','36','46','14','34','44','24',
    '22','32','42','12','17','27','37','47',
    '15','25','35','45','18','28','38','48'
    ]


    for (x, y, bw, bh), conf, c in zip(boxes, confs, cls):
        detections.append((x, y, bw, bh, float(conf), int(names[int(c)])))  # cls is already FDI

    quadrant_labels = {"UR": [], "UL": [], "LL": [], "LR": []}

    for (x, y, bw, bh, conf, fdi) in detections:
        cv2.circle(img, (int(x), int(y)), 5, (0, 0, 0), -1)   # black border
        cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1) # red dot

        cv2.putText(
              img,
              str(fdi),
              (int(x) - 10, int(y) + 25),  
              cv2.FONT_HERSHEY_SIMPLEX,
              0.5,                          
              (0, 0, 255),             
              1,                      
              cv2.LINE_AA
            )

        if 11 <= fdi <= 18:
           quadrant_labels["UR"].append(str(fdi))
        elif 21 <= fdi <= 28:
           quadrant_labels["UL"].append(str(fdi))
        elif 31 <= fdi <= 38:
           quadrant_labels["LL"].append(str(fdi))
        elif 41 <= fdi <= 48:
            quadrant_labels["LR"].append(str(fdi))


    y_offset = 30
    for quadrant in ["UR", "UL", "LL", "LR"]:
        if quadrant_labels[quadrant]:
            draw_text_with_outline(
                img,
                f"{quadrant}: " + ", ".join(quadrant_labels[quadrant]),
                (10, y_offset),
                font_scale=1,
                text_color=(0, 255, 0)
            )
            y_offset += 40  # move down for next label

    cv2.imwrite(save_path, img)
    print(f"✅ Processed image saved at {save_path}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python app.py <image_path>")
    else:
        img_path = sys.argv[1]
        process_image(img_path, save_path="output.jpg")


