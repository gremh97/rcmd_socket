# RCMD (Remote Command)

RCMD is a command-line interface tool for managing and running models on remote boards.

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