import click
import json
import os
import paramiko, pgrep
from typing import List, Dict
from .client import Client

def run_server(board):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(
            hostname=board['ipAddr'], 
            username=board['username'], 
            password=board['password'] 
        )
        
        # checking server process
        stdin, stdout, stderr = ssh.exec_command('pgrep -f main.py')
        process_output = stdout.read().decode().strip()
        
        if process_output:
            click.echo(f"Script is already running with PID(s): {process_output}")
        else:
            # 스크립트가 실행 중이 아니면 실행
            click.echo("Script is not running. Starting script...")
            stdin, stdout, stderr = ssh.exec_command(f"cd {os.path.join(board['eval_dir'],'server_dir')} && python main.py -p {board['port']}")
            click.echo(f"Started script: {stdout.read().decode()}")
        
    except paramiko.SSHException as e:
        click.echo(f"Failed to execute server script: {e}", err=True)
        return
    finally:
        ssh.close()

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

    def list_models(self, board_id:str, task:str):
        board = next((b for b in self.boards if b['boardID'] == board_id), None)
        if not board:
            click.echo(f"Board with ID {board_id} not found", err=True)
            return
        # run_server(board)
        client = Client(server_ip=board['ipAddr'], port=int(board['port']))
        try:
            client.communicate(
                option="mlist",
                task=task,
                eval_dir=board['eval_dir']
            )
        except Exception as e:
            click.echo(f"An error occurred: {e}", err=True)
        finally:
            client.communicate(option="exit")


    def run_model(self, board_id: str, model_name: str, task:str, lne:str, images:int, input_size:int):
        board = next((b for b in self.boards if b['boardID'] == board_id), None)
        if not board:
            click.echo(f"Board with ID {board_id} not found", err=True)
            return
        # run_server(board)
        client = Client(server_ip=board['ipAddr'], port=int(board['port']))
        try:
            client.communicate(
                option="test",
                task=task,
                model_name=model_name,
                lne_path=lne,
                num_images=images,
                preproc_resize=(input_size, input_size),
                log=False,
                eval_dir=board['eval_dir']
            )
        except Exception as e:
            click.echo(f"An error occurred: {e}", err=True)
        finally:
            client.communicate(option="exit")
    



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
@click.option('-b', '--board', required=True, help='(required) Board ID for run command')
@click.option('-t', '--task',  required=False, default= None, help='Task that model performs (classify, detect, detect_yolo)')
def mls(board, task):
    """List available models in the board"""
    rcmd = RCMD()
    rcmd.list_models(board_id=board, task=task)

@cli.command()
@click.option('-b', '--board', required=True, help='(required)Board ID for run command')
@click.option('-m', '--model', required=True, help='(required)Model name for run command')
@click.option('-t', '--task',  required=False, default= None, help='Task that model performs (classify, detect, detect_yolo)')
@click.option('-l', '--lne', required=False, default=None, help='path to .lne file in the board. `task` must be specified')
@click.option('-n', '--images', required=False, type=int, default=1000, help='Number of images for run command')
@click.option('-is', '--input_size', required=False, type=int, default=256, help='Input image size of model for run command')
def run(board, model, task, lne, images, input_size):
    """Run a model on a specific board"""
    rcmd = RCMD()
    rcmd.run_model(board_id=board, model_name=model, task=task, lne=lne, images=images, input_size=input_size)

if __name__ == "__main__":
    cli()