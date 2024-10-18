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

- To run a model on a specific board:
  ```
  rcmd run -b <board_id> -n <model_name>
  ```

For more information on each command, use the `--help` option:

```
rcmd --help
rcmd bls --help
rcmd run 