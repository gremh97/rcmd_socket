import click
import json
import os
import paramiko, time
from typing import List, Dict
from .client import Client

def wait_for_server_ready(ssh, port: int, max_retries: int = 10, retry_delay: float = 1.0) -> bool:
    """
    Wait for the server to be ready
    
    Args:
        ssh: SSH client instance
        port: Port number to check
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds
    
    Returns:
        bool: True if server is ready, False otherwise
    """
    for i in range(max_retries):
        # 1. Check if the process is running
        stdin, stdout, stderr = ssh.exec_command(f"pgrep -f 'python main.py -p {port}'")
        if not stdout.read().decode().strip():
            if i == max_retries - 1:
                click.echo("Server process not found after maximum retries", err=True)
                return False
            time.sleep(retry_delay)
            continue
        
        # 2. Check if the port is actually listening
        stdin, stdout, stderr = ssh.exec_command(f"netstat -tuln | grep {port}")
        if not stdout.read().decode().strip():
            if i == max_retries - 1:
                click.echo("Server port not listening after maximum retries", err=True)
                return False
            time.sleep(retry_delay)
            continue
            
        click.echo(f"Server is ready (attempt {i + 1}/{max_retries})")
        return True
        
    return False

def run_server(board) -> bool:
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        ssh.connect(
            hostname=board['ipAddr'], 
            username=board['username'], 
            password=board['password']
        )
        
        # Check server process
        stdin, stdout, stderr = ssh.exec_command(f"pgrep -f 'python main.py -p {board['port']}'")
        process_output = stdout.read().decode().strip()
        
        if process_output:
            click.echo(f"Script is already running with PID(s): {process_output}")
            # Even if already running, verify the server is responding
            if not wait_for_server_ready(ssh, board['port']):
                click.echo("Existing server is not responding properly", err=True)
                return False
        else:
            # If script is not running, start it
            click.echo("Script is not running. Starting script...")
            stdin, stdout, stderr = ssh.exec_command(
                f"cd {os.path.join(board['eval_dir'],'server_dir')} && "
                f"nohup python main.py -p {board['port']} > server.log 2>&1 &"
            )
            
            # Wait for server to be ready
            if not wait_for_server_ready(ssh, board['port']):
                click.echo("Failed to start server properly", err=True)
                return False
                
            click.echo("Server started successfully")
            
        return True
        
    except paramiko.SSHException as e:
        click.echo(f"Failed to execute server script: {e}", err=True)
        return False
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
        if not run_server(board): 
            return
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
        if not run_server(board): 
            return
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