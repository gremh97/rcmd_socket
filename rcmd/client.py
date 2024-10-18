import socket
import json
import click

class Client:
    def __init__(self, server_ip='192.168.0.163', port=5000, buffer_size=4096, timeout=None):
        self.server_ip = server_ip
        self.port = port
        self.buffer_size = buffer_size
        try:
            self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            click.echo("Socket successfully created")
        except socket.error as err:
            click.echo(f"socket creation failed with error {err}", err=True)

        self.client_socket.settimeout(timeout)
        self.client_socket.connect((self.server_ip, self.port))
        click.echo(f"Connected to server at {self.server_ip}:{self.port}")

    def send_data(self, data):
        serialized_data = json.dumps(data)
        self.client_socket.sendall(serialized_data.encode('utf-8'))

    def receive_data(self):
        response = b""
        while True:
            try:
                chunk = self.client_socket.recv(self.buffer_size)
                if not chunk:
                    break
                response += chunk
                if response.endswith((b'}',b'$')): # Assuming JSON response
                    break
            except socket.timeout:
                click.echo("Timeout while receiving data", err=True)
                break
        return json.loads(response.decode('utf-8'))

    def communicate(self, task, model_name=None, lne_path=None, num_images=1000, preproc_resize=(256, 256), log=False):
        if task == "exit":
            exit_msg = {"task":"exit"}
            self.send_data(exit_msg)
            click.echo("Exit message sent. Closing connection.")
            try:
                response = self.client_socket.recv(self.buffer_size).decode('utf-8')
                click.echo(response)
            except:
                click.echo("No response from server (likely already closed).")
            self.client_socket.close()
            return

        if not(model_name or lne_path):
            raise ValueError("One of model_name or lne_path must be provided")
        params = {
            "task": task,
            "model_name": model_name,
            "lne_path": lne_path,
            "num_images": num_images,
            "preproc_resize": preproc_resize,
            "log": log
        }

        # Send request
        self.send_data(params)
        click.echo("Model Evaluation request sent. Waiting for response...")

        # Receive result
        result = self.receive_data()
        self.display_result(result)
        if log: 
            self.save_log_data(result.get('model_name', 'unknown_model'), result.get('log_data', {}))

    def display_result(self, result):
        filtered_result = {}
        for k, v in result.items():
            if k == 'log_data':
                continue
            try:
                numeric_value = float(v)
                filtered_result[k] = round(numeric_value, 2)
            except (ValueError, TypeError):
                filtered_result[k] = v

        click.echo("Received result from server:")
        click.echo(json.dumps(filtered_result, indent=4))

    def save_log_data(self, model_name, log_data):
        log_path = f"output/{model_name}.log"
        with open(log_path, 'w') as json_file:
            json.dump(log_data, json_file, indent=4)
        click.echo(f"log_data saved to {log_path}")