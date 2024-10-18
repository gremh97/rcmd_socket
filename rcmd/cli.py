import click
import json
import os
from typing import List, Dict
from .client import Client

class RCMD:
    def __init__(self):
        self.boards = self.load_boards()

    def load_boards(self) -> List[Dict]:
        script_dir = os.path.dirname(os.path.realpath(__file__))
        boards_path = os.path.join(script_dir, 'boards.json')
        with open(boards_path, 'r') as f:
            data = json.load(f)
        return data['boards']

    def list_boards(self):
        for board in self.boards:
            click.echo(f"Board ID: {board['boardID']}, IP: {board['ipAddr']}, Port: {board['port']}")

    def run_model(self, board_id: str, model_name: str, images:int, input_size:int):
        board = next((b for b in self.boards if b['boardID'] == board_id), None)
        if not board:
            click.echo(f"Board with ID {board_id} not found", err=True)
            return

        client = Client(server_ip=board['ipAddr'], port=int(board['port']))
        try:
            client.communicate(
                task="classify",
                model_name=None,
                lne_path=f"{board['eval_dir']}classify/{model_name}.lne",
                num_images=images,
                preproc_resize=(input_size, input_size),
                log=False
            )
        except Exception as e:
            click.echo(f"An error occurred: {e}", err=True)
        finally:
            client.communicate(task="exit")

@click.group()
def cli():
    """Remote Command (RCMD) CLI"""
    pass

@cli.command()
def bls():
    """List available boards"""
    rcmd = RCMD()
    rcmd.list_boards()

@cli.command()
@click.option('-b', '--board', required=True, help='Board ID for run command')
@click.option('-m', '--model', required=True, help='Model name for run command')
@click.option('-n', '--images', required=False, type=int, default=100, help='Number of images for run command')
@click.option('-is', '--input_size', required=False, type=int, default=256, help='Input image size of model for run command')
def run(board, model, images, input_size):
    """Run a model on a specific board"""
    rcmd = RCMD()
    rcmd.run_model(board, model, images, input_size)

if __name__ == "__main__":
    cli()