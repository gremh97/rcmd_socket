#!/usr/bin/env python3

import argparse
import lne_tflite.interpreter as lne
import PIL.Image
import numpy as np
import glob
from tqdm.contrib import tenumerate
import time
import json


class ClassifierEvaluator:
    def __init__(self, model_name=None, lne_path="/home/aimf/evaluate/classify/efficientnet_lite0.lne", 
                 num_images=1000, preproc_resize=(256,256), log=False):
        self.model_name         = model_name or lne_path.split("/")[-1].split(".")[0]
        self.lne_path           = lne_path
        self.preproc_fn     = self._get_preproc_fn()
        self.num_images         = num_images
        self.preproc_resize     = preproc_resize
        self.log                = log
        self.log_data           = []

        #  Model Initalization
        self.model              = lne.Interpreter(self.lne_path)
        self.model.allocate_tensors()
        self.input_shape        = self.model.get_input_details()[0]["shape"][1:3]
        self.output_detail      = self.model.get_output_details()[0]
        self.groundtruth        = [int(_) for _ in open("/home/aimf/evaluate/classify/labels/groundtruth.txt").readlines()]
    

    def _get_preproc_fn(self):
        if   self.model_name.startswith(("mobilenetv1", "efficientnet")):
            return lambda x:x/127.5-1
        elif self.model_name.startswith("mobilenetv4"):
            return lambda x:(x/255-[0.485,0.456,0.406])/[0.229,0.224,0.225]
        elif self.model_name.startswith("vgg"):
            return lambda x: x-[123.68,116.779,103.939]
        else: raise NotImplementedError(f"Invalid model_name={self.model_name} provided")


    def evaluate(self):
        print(f"... {self.model_name} evaluation in progress...")
        (IH, IW)        = self.input_shape
        (RH, RW)        = self.preproc_resize
        crop_box        = [(RW - IW) // 2, (RH - IH) // 2, (RW - IW) // 2 + IW, (RH - IH) // 2 + IH]

        # image_files = sorted(glob.glob(f"/dataset/imagenet/validate_5000/*"))
        # image_files     = sorted(glob.glob(f"/data/image/imagenet/validate_5000/*"))
        image_dirs      = [
            "/data/imagenet/ILSVRC2012_img_val_1",
            "/data/imagenet/ILSVRC2012_img_val_2",
            "/data/imagenet/ILSVRC2012_img_val_3",
            "/data/imagenet/ILSVRC2012_img_val_4",
            "/data/imagenet/ILSVRC2012_img_val_5",
        ]
        image_files     = []
        for image_dir in image_dirs:
            image_files.extend(sorted(glob.glob(f"{image_dir}/*")))
        fps             = []
        top1, top5      = 0, 0

        for n, image_file in tenumerate(image_files[:self.num_images]):
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
            "model_name"    : self.model_name,
            "lne_path"      : self.lne_path,
            "average_fps"   : f"{np.mean(fps)}",
            "min_fps"       : f"{np.min(fps)}",
            "max_fps"       : f"{np.max(fps)}",
            "top1_accuracy" : f"{top1 / self.num_images}",
            "top5_accuracy" : f"{top5 / self.num_images}",
            "log_data"      : self.log_data
        }

        return results
    

if __name__ == "__main__":
    evaluator = ClassifierEvaluator(num_images=50000)
    results = evaluator.evaluate()
    print(results)