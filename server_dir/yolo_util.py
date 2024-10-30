import numpy as np
import os
import pathlib
from dataclasses import dataclass


anchors_v2      = [
  [ [  9.1637,  10.8382],
    [ 29.9914,  33.0005],
    [ 53.4149,  87.5894],
    [126.1251,  56.4445],
    [156.3283, 146.6925] ]      # large  box
]
anchors_v3tiny  = [
  [ [ 40.5,     41.0   ],
    [ 67.5,     84.5   ],
    [172.0,    159.5   ] ],     # large  box
  [ [  5.0,      7.0   ],
    [ 11.5,     13.5   ],
    [ 18.5,     29.0   ] ]      # medium box
]
anchors_v3      = [
  [ [ 58.0,     45.0   ],
    [ 78.0,     99.0   ],
    [186.5,    162.0   ] ],     # large  box
  [ [ 15.0,     30.5   ],
    [ 31.0,     22.5   ],
    [ 29.5,     59.5   ] ],     # medium box
  [ [  5.0,      6.5   ],
    [  8.0,     15.0   ],
    [ 16.5,     11.5   ] ]      # small  box
]

def _decode_boxenc(anchor, box, boxid, grid=(13,13), sample=32):
  a       =  boxid% len(anchor)
  cx      = (boxid//len(anchor))% grid[1]
  cy      = (boxid//len(anchor))//grid[1]

  w_2     = np.exp(box[0])*anchor[a][0]
  h_2     = np.exp(box[1])*anchor[a][1]
  x       = (cx +  box[2])*sample
  y       = (cy +  box[3])*sample
  box[0]  = x-w_2       # upper-left  x
  box[1]  = y-h_2       # upper-left  y
  box[2]  = x+w_2       # lower-right x 
  box[3]  = y+h_2       # lower-right y

def _process_iou(boxes, classes, scores, thres_iou):
  nboxes      = len(scores)
  orders      = np.argsort(scores)[::-1]

  ##  Find the area of boxes.
  boxes_area  = np.ndarray(nboxes, dtype=np.float32)
  valids      = np.ndarray(nboxes, dtype=int)
  for i, box in enumerate(boxes):
    box_w       = box[2]-box[0]
    box_h       = box[3]-box[1]
    valids[i]   = (box_w>0) and (box_h>0)
    if (valids[i]): boxes_area[i] = box_w*box_h

  ##  IOU operation.
  for i, id_i in enumerate(orders):
    if not(valids[id_i]):  continue

    for id_j in orders[i+1:]:
      if (not(valids[id_j]) or classes[id_i]!=classes[id_j]): continue

      x1        = max(boxes[id_i][0], boxes[id_j][0])
      y1        = max(boxes[id_i][1], boxes[id_j][1])
      x2        = min(boxes[id_i][2], boxes[id_j][2])
      y2        = min(boxes[id_i][3], boxes[id_j][3])
      if (x2<=x1 or y2<=y1):  continue

      area_ixj  = (x2-x1)*(y2-y1)
      area_iuj  = boxes_area[id_i]+boxes_area[id_j]-area_ixj
      if (area_ixj>=thres_iou*area_iuj):  valids[id_j] = 0

  ##  Detection outputs.
  detects     = []
  for id_i in orders:
    if (valids[id_i]):   detects.append(id_i)

  return [
    len(detects),                       # number of detections
    np.asarray(classes)[detects],       # detection classes
    np.asarray(scores )[detects],       # detection scores
    np.asarray(boxes  )[detects]        # detection boxes
  ]

def postprocess(
  network,
  preds0,               # w, h
  preds1,               # x, y, object, classes
  image_shape=(416,416), thres_score=0.30, thres_iou=0.50
):
  if   (network=="yolov2" or network=="yolov2_tiny"):
    pyramids  = 1
    anchors   = anchors_v2
  elif (network=="yolov3"):     # [AiMF-20230612-kwangmo]
    pyramids  = 2
    anchors   = anchors_v3tiny
  else:
    print("[ERROR] Unknown network \"%s\"." %(network))
    exit()

  ##  Filter by class scores.
  boxes     = []
  classes   = []
  scores    = []

  for p in range(pyramids):
    if (p==0):
      sample    = 32
      grid_h    = image_shape[0]//sample
      grid_w    = image_shape[1]//sample
      nboxes    = grid_h*grid_w*len(anchors[0])
      boxid0    = 0
      boxidN    = nboxes
    else:
      sample   /= 2
      grid_h   *= 2
      grid_w   *= 2
      nboxes   *= 4
      boxid0    = boxidN
      boxidN   += nboxes

    for boxid in range(boxid0, boxidN):
      pred0     = preds0[boxid]
      pred1     = preds1[boxid]
      classid   = np.argmax(pred1[3:])
      score     = pred1[2]*pred1[3+classid]
      box       = np.hstack((pred0,pred1[0:2]))

      if (score>=thres_score):
        _decode_boxenc(anchors[p], box, boxid-boxid0, (grid_h,grid_w), sample)
        boxes  .append(box    )
        classes.append(classid)
        scores .append(score  )

  ##  IOU operation
  return _process_iou(boxes, classes, scores, thres_iou)
