# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
from ultralytics.utils.plotting import Annotator
from utils.torch_utils import select_device, smart_inference_mode
from utils.general import (
    LOGGER, Profile, check_file, check_img_size, check_imshow,
    check_requirements, colorstr, cv2, increment_path, non_max_suppression,
    print_args, scale_boxes, strip_optimizer, xyxy2xywh
)
from utils.dataloaders import LoadStreams
from models.common import DetectMultiBackend
import os
import torch
import sys
from PIL import ImageFilter, Image
from tools import transform_color
import array
import pyttsx3 as tts
import numpy as np
import cv2
import threading

# Global Variables
ROOT = os.path.dirname(os.path.abspath(__file__))
weights = 'best3.pt'  # os.path.join(ROOT, 'yolov5s.pt')  #
source = 1  # Use webcam
data = 'data.yaml'  # os.path.join(ROOT, 'data/coco128.yaml')  #
imgsz = (256, 256)
conf_thres = 0.35
iou_thres = 0.25
max_det = 1000
device =  'cpu' # Use CPU
view_img = True
classes = None
agnostic_nms = False
augment = False
visualize = False
line_thickness = 0.5
hide_conf = False
half = False
dnn = True
vid_stride = 1
counter = 0
statement = ""
stop_thread = False
check_thread = None
# Variable Distance for Zoning
Zone_1 = 100
Zone_2 = 200
Zone_3 = 300
focal_length = 19
distance = 0
actual_width = 50
# Varable color for glow effect
BG_COLOR = '000000'
GLOW_COLOR = 'ec0e77'  # (0.929, 0.055, 0.467)
FG_COLOR_1 = 'ff31f4'  # (1, 0.196, 0.957)
FG_COLOR_2 = 'ffd796'  # (1, 0.847, 0.592)
FILL_COLOR = 'FFFFFF'
# Priority colors
PRIORITY_COLORS = {
    1: (0, 0, 255),    # Red
    2: (0, 165, 255),  # Orange
    3: (0, 255, 255),  # Yellow
    4: (0, 255, 0)     # Green
}


# FILE = os.path.join(ROOT, 'yolov5s.pt')  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = os.path.relpath(ROOT, os.path.join(os.getcwd()))  # relative

# ==========================Just to decorate the display and for fun=================================================


def create_glowing_neon(image, rect_position, rect_size, rect_color, text):
    # Create an image with the same size as the input image
    neon_image = np.zeros_like(image)

    # Draw a filled rectangle with a neon effect
    cv2.rectangle(neon_image, rect_position, rect_size, rect_color, cv2.FILLED)

    # Apply a larger Gaussian blur to the neon rectangle
    neon_image = cv2.GaussianBlur(neon_image, (0, 0), 20)

    # Create a mask for the original rectangle
    rect_mask = np.zeros_like(image)
    cv2.rectangle(rect_mask, rect_position, rect_size,
                  (255, 255, 255), cv2.FILLED)

    # Add the original rectangle to the neon effect image
    glowing_neon = cv2.addWeighted(image, 1, neon_image, 0.8, 0)

    # Draw the text inside the rectangle
    cv2.putText(glowing_neon, text, (rect_position[0] + 10, rect_position[1] + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, rect_color, 2, cv2.LINE_AA)

    return glowing_neon


@smart_inference_mode()
def run():
    global weights, source, data, imgsz, conf_thres, iou_thres, max_det, device, view_img, classes, agnostic_nms, augment, visualize, line_thickness, hide_conf, half, dnn, vid_stride, counter,statement,check_thread

    source = str(source)
    webcam = isinstance(source, int) or source.isdigit()

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(
        weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz,
                              stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(
                pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # ===================================================================================================================
        # ===========Distance calculation and declare zone based on width of detected object=================================
        # ===================================================================================================================

        # Initialize variables for highest priority and alarm
        highest_priority = 4
        alarm_triggered = False
        alarm_object_details = None  # Store details of the object triggering the alarm

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1

            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '

            annotator = Annotator(
                im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(
                    im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "

                # Write results
                for j, (*xyxy, conf, cls) in enumerate(reversed(det)):
                    c = int(cls)  # integer class

                    width = xyxy[2] - xyxy[0]
                    height = xyxy[3] - xyxy[1]
                    actual_width = 50

                    # Process different classes===========================================================================
                    if names[c] == 'box':
                        actual_width = 100
                    elif names[c] == 'cone':
                        actual_width = 50
                    elif names[c] == 'drum':
                        actual_width = 60
                    elif names[c] == 'ibc':
                        actual_width = 100
                    elif names[c] == 'face':
                        actual_width = 12
                    elif names[c] == 'forklift':
                        actual_width = 100
                    elif names[c] == 'person':
                        actual_width = 50

                    # Calculate distance, 0.02625 is the pixel width at 1280x720 resolution
                    distance = (actual_width * focal_length) / \
                        (width * 0.02625)

                    # Split distance into zones
                    if distance < Zone_1:
                        zone = 'Stop Zone'
                        priority = 1
                    elif Zone_1 <= distance < Zone_2:
                        zone = 'Danger Zone'
                        priority = 2
                    elif Zone_2 <= distance < Zone_3:
                        zone = 'Warning Zone'
                        priority = 3
                    else:
                        zone = 'Safe Zone'
                        priority = 4

                    conf_percent = conf * 100  # Convert confidence to percentage
                    label = f'{names[c]} {conf_percent:.0f}% , W:{width:.0f}, H:{height:.0f}, Dist:{distance:.2f}cm, Zone:{zone}'
                    annotator.box_label(xyxy, label, color=(16776960, True))

                    # Update highest priority
                    if priority < highest_priority:
                        highest_priority = priority
                        alarm_triggered = True
                        alarm_object_details = f'{names[c]} {j} - Priority: {priority}, Zone: {zone}, Dist: {distance:.2f}cm'

                    # Draw alarm information at the right top of the frame
                    if alarm_triggered == True:
                        alarm_text = f'ALERT - {alarm_object_details}'
                        text_size = 0.3
                        text_thickness = 1

                        # Calculate rectangle size
                        rect_width = 410
                        rect_height = 20
                        rect_x = im0.shape[1] - rect_width - \
                            10  # 10 pixels from the right
                        rect_y = 10

                        # Get priority color
                        priority_color = PRIORITY_COLORS[highest_priority]
                        rect_color = (*priority_color, 50)
                        cv2.rectangle(im0, (rect_x, rect_y), (rect_x + rect_width,
                                                              rect_y + rect_height), rect_color, cv2.FILLED)

                        cv2.putText(im0, alarm_text, (rect_x + 2, rect_y + 12), cv2.FONT_HERSHEY_SIMPLEX,
                                    text_size, (255, 255, 255, 50), text_thickness, cv2.LINE_AA)
                        
                        counter = counter + 1
                        print(counter)
                        
                        if counter == 50:
                            if highest_priority == 1:
                                statement = "STOP"
                            elif highest_priority == 2:
                                statement = "DANGER"
                            elif highest_priority == 3:
                                statement = "CAUTION"
                
                            check_thread = play_tts(statement)
                            counter = 0

            # Stream results
            im0 = annotator.result()
            scale_percent = 160  # percent of original size
            width = int(im0.shape[1] * scale_percent / 100)
            height = int(im0.shape[0] * scale_percent / 100)
            dim = (width, height)

            # resize image
            resized = cv2.resize(im0, dim, interpolation=cv2.INTER_AREA)
            cv2.imshow("Object detection", resized)

    cv2.waitKey(1)  # 1 millisecond
    cv2.destroyAllWindows()

def play_tts(statement):
    global stop_thread
    
    stop_thread = False
    
    engine = tts.init()
    engine.say(statement)
    
    thread = threading.Thread(target=engine.runAndWait)
    thread.start()
    
    return thread

def stop_tts(thread):
    global stop_thread
    
    stop_thread = True
    
    thread.join()
    print("Thread stopped")
    
def main():
    global check_thread
    
    run()
    stop_tts(check_thread)

if __name__ == '__main__':
    main()
