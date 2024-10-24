# RCMD (Remote Command)

RCMD is a command-line interface tool for managing and running models on remote boards.

## Preset
`server_dir` should be located in proper directory,
inside of board's `eval_dir`, before Installation and Run `rcmd` command. 
Each model should be under `{classify, detect, detect_yolo}/input`

```
eval_dir = /home/aimf/evaluate

*
└── evaluate
    ├── classify
    │   ├── input
    │   │   └── efficientnet_lite0.lne
    │   ├── labels
    │   │   ├── category.txt
    │   │   └── groundtruth.txt
    │   ├── output
    │   │   └── mobilenetv1.lne
    │   ├── test_classify.py
    ├── detect
    │   ├── input
    │   │   ├── yolov2_tiny.lne
    │   │   ├── yolov3_keti.lne
    │   │   └── yolov3_tiny.lne
    │   ├── objectdetect_yolo.py
    │   └── output
    │       └── yolov3.json
    ├── detect_yolo
    │   ├── export_json.sh
    │   ├── input
    │   │   └── yolov2.lne
    │   ├── labels
    │   │   ├── coco80.txt
    │   │   └── coco90.txt
    │   ├── test_detect.py
    │   ├── test_detect_yolo.py
    │   ├── test_yolo.sh
    │   ├── yolo_util.py
    │   ├── yolov2.json
    │   └── yolov2.lne
    └── server_dir
        ├── ModelEvaluator.py
        ├── json_utils.py
        ├── main.py
        ├── models_by_task.json
        ├── remote_server.py
        └── server.py
```


## Installation

To install RCMD, run the following command:

```
pip install .
```

## Usage

After installation, you can use the `rcmd` command directly from your terminal:

- To list available boards:
  ```
  rcmd bls
  ```

- To list available models in the board
  ```
  rcmd mls -b <board_id>
  ```

- To run a model on a specific board:
  ```
  rcmd run -b <board_id> -m <model_name>
  ```
  - options
    `-b, --board`     : board id **required**
    `-m, --model`     : model name **required**
    `-t, --task`      : taks(classify, detect, detect_yolo)
    `-l, --lne`       : lne path in board (When assigning the `lne`, the `task` must be explicitly specified.)
    `-n, --images`    : number of images (default=1000)
    `-is, input_size` : Input image size of model (default: 256)
    ``

For more information on each command, use the `--help` option:

```
rcmd --help
rcmd bls --help
rcmd run 

