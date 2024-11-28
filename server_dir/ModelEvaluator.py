#!/usr/bin/env python3

import tensorflow as tf
import PIL.Image
import numpy as np
import glob
from tqdm.contrib import tenumerate
import time
import json
import os

class ClassifierEvaluator:
    def __init__(self, model_name=None, tflite_path="/home/evaluate/classify/input/efficientnet_lite0.tflite", 
                 num_images=1000, preproc_resize=(256,256), log=False):
        self.model_name         = model_name or tflite_path.split("/")[-1].split(".")[0]
        self.tflite_path           = tflite_path
        self.preproc_fn         = self._get_preproc_fn()
        self.num_images         = num_images
        self.preproc_resize     = preproc_resize
        self.log                = log
        self.log_data           = []
        self.eval_dir           = os.path.dirname(os.path.dirname(tflite_path))

        #  Model Initalization
        self.model              = tf.lite.Interpreter(self.tflite_path)
        self.model.allocate_tensors()
        self.input_shape        = self.model.get_input_details()[0]["shape"][1:3]
        self.output_detail      = self.model.get_output_details()[0]
        self.groundtruth        = [int(_) for _ in open(os.path.join(self.eval_dir, "labels/groundtruth.txt")).readlines()]
    

    def _get_preproc_fn(self):
        if   self.model_name.startswith(("mobilenetv1", "efficientnet")):
            return lambda x:x/127.5-1
        elif self.model_name.startswith("mobilenetv4"):
            return lambda x:(x/255-[0.485,0.456,0.406])/[0.229,0.224,0.225]
        elif self.model_name.startswith("vgg"):
            return lambda x: x-[123.68,116.779,103.939]
        else: raise NotImplementedError(f"Invalid model_name={self.model_name} provided")


    def evaluate(self, client_socket=None):
        print(f"... {self.model_name} evaluation in progress...")
        (IH, IW)        = self.input_shape
        (RH, RW)        = self.preproc_resize
        crop_box        = [(RW - IW) // 2, (RH - IH) // 2, (RW - IW) // 2 + IW, (RH - IH) // 2 + IH]

        # image_files = sorted(glob.glob(f"/dataset/imagenet/validate_5000/*"))
        # image_files     = sorted(glob.glob(f"/data/image/imagenet/validate_5000/*"))
        image_dirs      = [
            "/data/image/imagenet/ILSVRC2012_img_val_1",
            "/data/image/imagenet/ILSVRC2012_img_val_2",
            "/data/image/imagenet/ILSVRC2012_img_val_3",
            "/data/image/imagenet/ILSVRC2012_img_val_4",
            "/data/image/imagenet/ILSVRC2012_img_val_5",
        ]
        image_files     = []
        for image_dir in image_dirs:
            image_files.extend(sorted(glob.glob(f"{image_dir}/*")))
        fps             = []
        top1, top5      = 0, 0
        start_t = time.time()

        for n, image_file in tenumerate(image_files[:self.num_images]):
            if client_socket: 
                current_fps = (n + 1) / (time.time() - start_t) if time.time() - start_t > 0 else 0
                current = n+1
                total = self.num_images
                progress = {
                "type": "progress",
                "current": current,
                "total": total,
                "progress": f"{(current/total*100):.1f}%",
                "speed": f"{current_fps:.2f}it/s"}
                
                try:
                    client_socket.send((json.dumps(progress)+"\n").encode('utf-8'))
                except:
                    print("Failed to send progress")


            image       = PIL.Image.open(image_file).convert("RGB").resize(self.preproc_resize).crop(crop_box)
            inputs      = self.preproc_fn(np.expand_dims(np.asarray(image), 0)).astype("float32")
            self.model  .set_tensor(self.model.get_input_details()[0]["index"], inputs)
            t0          = time.time()
            self.model  .invoke()
            fps         .append(1 / (time.time() - t0))

            outputs     = np.ravel(self.model.get_tensor(self.output_detail["index"]))
            predicts5   = outputs.argsort()[[-1, -2, -3, -4, -5]]
            top5        += self.groundtruth[n] in predicts5
            top1        += self.groundtruth[n] == predicts5[0]

            if self.log:
                log_entry = f"{image_file.split('/')[-1]}   {predicts5[0]}  {self.groundtruth[n]}   " \
                            f"{'PASS' if (self.groundtruth[n] == predicts5[0]) else 'FAIL'}  " \
                            f"{'PASS' if (self.groundtruth[n] in predicts5) else 'FAIL'} {predicts5}\n"
                self.log_data.append(log_entry)

        results = {
            "type"          : "complete_classifier",
            "model_name"    : self.model_name,
            "tflite_path"      : self.tflite_path,
            "average_fps"   : f"{np.mean(fps)}",
            "min_fps"       : f"{np.min(fps)}",
            "max_fps"       : f"{np.max(fps)}",
            "top1_accuracy" : f"{top1 / self.num_images}",
            "top5_accuracy" : f"{top5 / self.num_images}",
            "log_data"      : self.log_data
        }

        return results
    
##  End of class ClassifierEvaluator



import random
import colorsys
import cv2
from yolo_util import postprocess

def generate_colors(nclasses):
    colors    = []
    for i in range(nclasses):
        r,g,b     = colorsys.hsv_to_rgb(i/nclasses,1,1)
        colors.append([int(255*r),int(255*g),int(255*b)])

    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    return colors

class YoloEvaluator:
    def __init__(self, model_name, tflite_path, num_images=5000, save_json=False):
        self.model_name     = model_name
        self.num_images     = num_images       
        self.interpreter    = tf.lite.Interpreter(tflite_path)
        self.interpreter.allocate_tensors()
        self.input_detail   = self.interpreter.get_input_details()
        self.output_detail  = self.interpreter.get_output_details()

        self.height,self.width  = self.input_detail[0]['shape'][1:3]
        self.square         = (self.height==self.width)
        self.class_names    = [
            "person",     "bicycle",    "car",        "motorbike",  "aeroplane",
            "bus",        "train",      "truck",      "boat",       "traffic light",
            "fire hydrant", "stop sign","parking meter", "bench",   "bird",
            "cat",        "dog",        "horse",      "sheep",      "cow",
            "elephant",   "bear",       "zebra",      "giraffe",    "backpack",
            "umbrella",   "handbag",    "tie",        "suitcase",   "frisbee",
            "skis",       "snowboard",  "sports ball", "kite",      "baseball bat",
            "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "wine glass", "cup",        "fork",       "knife",      "spoon",
            "bowl",       "banana",     "apple",      "sandwich",   "orange",
            "broccoli",   "carrot",     "hot dog",    "pizza",      "donut",
            "cake",       "chair",      "sofa",       "pottedplant","bed",
            "diningtable","toilet",     "tvmonitor",  "laptop",     "mouse",
            "remote",     "keyboard",   "cell phone", "microwave",  "oven",
            "toaster",    "sink",       "refrigerator", "book",     "clock",
            "vase",       "scissors",   "teddy bear", "hair drier", "toothbrush"
        ]
        self.colors         = generate_colors(len(self.class_names))
        self.save_json      = save_json
        self.fps            = []

    def _preprocess(self, img, fill=128):
        img         = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if (self.square):
            ##  Keep original image proportion and resize
            h, w, _     = img.shape
            squareside  = self.height
            fit_ratio   = min(squareside/h, squareside/w)
            fit_h       = int(h*fit_ratio)
            fit_w       = int(w*fit_ratio)
            if max(fit_h, fit_w) != 416:
                if fit_h == max(fit_h, fit_w):
                    fit_h = 416
                elif fit_w == max(fit_h, fit_w):
                    fit_w = 416
            fit_img     = cv2.resize(img, (fit_w, fit_h))

            ##  Make image with average-padding.
            img         = np.full((squareside,squareside,3), fill, img.dtype)
            if   (fit_h<fit_w):
                img[(squareside-fit_h)//2:(squareside+fit_h)//2,:,:]  = fit_img
            elif (fit_w<fit_h):
                img[:,(squareside-fit_w)//2:(squareside+fit_w)//2,:]  = fit_img
        else:
            img         = cv2.resize(img, (self.width,self.height))

        return np.expand_dims(img.astype(np.float32),axis=0)/256
    ##  End of preprocess()

    def _inference(self, img):
        self.interpreter.set_tensor(self.input_detail[0]['index'], img)
        t0          = time.time()
        self.interpreter.invoke()
        self.fps.append(1 / (time.time() - t0))

        OD        = self.output_detail      # shorcut to output_detail
        if   ("yolov2" in self.model_name):
            preds0  = self.interpreter.get_tensor(OD[1]["index"]).reshape((-1, 2))
            preds1  = self.interpreter.get_tensor(OD[0]["index"]).reshape((-1,83))
            thres_score = 0.40
        elif ("yolov3" in self.model_name):
            preds0  = np.concatenate([
                self.interpreter.get_tensor(OD[3]["index"]).reshape((-1, 2)),
                self.interpreter.get_tensor(OD[0]["index"]).reshape((-1, 2))
            ], axis=0)
            preds1  = np.concatenate([
                self.interpreter.get_tensor(OD[1]["index"]).reshape((-1,83)),
                self.interpreter.get_tensor(OD[2]["index"]).reshape((-1,83))
            ], axis=0)
            thres_score = 0.30
        return postprocess(self.model_name, preds0, preds1, (self.height,self.width), thres_score, thres_iou=0.60)
    
    def evaluate(self, client_socket=None):
        print(f"... {self.model_name} evaluation in progress...")
        check_set = set()
        class_names = self.class_names
        input_dir  = "/data/image/coco/val2017/"
        image_list = os.listdir(input_dir)
        
        detections = []
        start_t = time.time()
        
        for n, image_name in tenumerate(image_list[:self.num_images]):
            if client_socket: 
                current_fps = (n + 1) / (time.time() - start_t) if time.time() - start_t > 0 else 0
                current = n+1
                total = self.num_images
                progress = {
                "type": "progress",
                "current": current,
                "total": total,
                "progress": f"{(current/total*100):.1f}%",
                "speed": f"{current_fps:.2f}it/s"}
                
                try:
                    client_socket.send((json.dumps(progress)+"\n").encode('utf-8'))
                except:
                    print("Failed to send progress")
        
            image_path = f'{input_dir}{image_name}'
            image_id = image_name.split('.')[0]
            image = cv2.imread(image_path)
            
            input_img = self._preprocess(image)
            _, classes, scores, boxes = self._inference(input_img)
    

            if (self.square):
                for box in boxes:
                    box[1]  = 208+(box[1]-208)*4/3
                    box[3]  = 208+(box[3]-208)*4/3

            h, w, _     = image.shape
            h_ratio     = h/416.0
            w_ratio     = w/416.0

            for i,c in enumerate(classes):
                left    = max(0, int(np.round(boxes[i][0]*w_ratio)))
                top     = max(0, int(np.round(boxes[i][1]*h_ratio)))
                right   = min(w, int(np.round(boxes[i][2]*w_ratio)))
                bottom  = min(h, int(np.round(boxes[i][3]*h_ratio)))

                class_name  = class_names[c]
                score       = scores[i]

                content = f"{class_name} {score:.2f} {left:.2f} {top:.2f} {right:.2f} {bottom:.2f}"
                detections.append({
                    "image_id"    : int(image_id),
                    "category_id" : int(c)+1,
                    "bbox"        : [
                        float(left),
                        float(top),
                        float(right-left),
                        float(bottom-top)
                    ],
                    "score"       : float(score)
                })
        if self.save_json:
            detection_file = f"./output/{self.model_name}.json"
            open(detection_file,"w").write(json.dumps(detections, indent=2))
            print("End inferencing: JSON file is ready for AP evaluation.")
            print("Result")
            print(f"  Output JSON   : {detection_file}")
        
        results = {
            "type"        : "complete_yolo",
            "model_name"  : self.model_name,
            "average_fps" : np.mean(self.fps),
            "detections"  : detections
        }
        return results               
        
##  End class YoloEvaluator



    

if __name__ == "__main__":
    # evaluator = ClassifierEvaluator(num_images=50000)
    # results = evaluator.evaluate()
    # print(results)
    
    evaluator = YoloEvaluator(
        model_name="yolov2" , 
        tflite_path="/home/evaluate/detect_yolo/input/yolov2.tflite", 
        num_images=100, 
        save_json=True
    )
    results = evaluator.evaluate()
