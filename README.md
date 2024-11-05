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

## Usage via CLI

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
``` 



## Usage in Script

```python
from rcmd.cli import  RCMD

rcmd_instance = RCMD()

result1 = rcmd_instance.list_boards(return_output=True)                         # if you want to get result, set return_output=True
print(result1)
"""[{'boardID': '001', 'ipAddr': '192.168.0.163', 'port': '5000', 'username': 'aimf', 'password': 'siliconeuro', 'eval_dir': '/home/aimf/evaluate/'}, {'boardID': '002', 'ipAddr': '192.168.0.159', 'port': '5001', 'username': 'aimf', 'password': 'siliconeuro', 'eval_dir': '/home/aimf/evaluate/'}]"""

for board in result1:
    result2 = rcmd_instance.list_models(board_id=board.get('boardID'), 
                                        task=None,                               # task = (None, classify, detect, detect_yolo)
                                        return_output=True                       # if you want to get result, set return_output=True
                                        )    
    print(result2)
    """
    {'classify': {'dir': '/home/aimf/evaluate/classify/input', 'models': [{'name': 'mobilenetv4', 'hash': 'f8447c6f53ea4baf910b225792c3a716'}, {'name': 'mobilenetv1', 'hash': 'e45e6f07f31f0dfeb404d330a012c6bc'}]}, 'detect': {'dir': '/home/aimf/evaluate/detect/input', 'models': [{'name': 'yolov3_tiny', 'hash': '86929ba0d551e95f624608b358bec0b6'}, {'name': 'yolov2_tiny', 'hash': '0fe8890b3c964b76575ab765a1a2ffc7'}, {'name': 'yolov3_keti', 'hash': 'c2224dfe1d015545239c151f0fec4959'}]}, 'detect_yolo': {'dir': '/home/aimf/evaluate/detect_yolo/input', 'models': [{'name': 'yolov2', 'hash': 'c79e36b0a05856605ca1f70df2b21592'}]}}
    """ 


result3 = rcmd_instance.run_model(board_id='001', 
                                 model_name='efficientnet_lite0',
                                 task="classify",
                                 lne=None,
                                 images=100,
                                 input_size=256,
                                 return_output=True                             # if you want to get result, set return_output=True
                                 )
print(result3)

"""
{'type': 'complete_classifier', 'model_name': 'efficientnet_lite0', 'lne_path': '/home/aimf/evaluate/classify/input/efficientnet_lite0.lne', 'average_fps': '100.12584305059468', 'min_fps': '96.83706970193707', 'max_fps': '100.86098352771432', 'top1_accuracy': '0.72', 'top5_accuracy': '0.9', 'log_data': []}
"""
```

