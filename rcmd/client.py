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
        try:
            # Check if JSON 
            return json.loads(response.decode('utf-8'))
        except json.JSONDecodeError:
            return response.decode('utf-8')

    def communicate(self, option, task=None, model_name=None, lne_path=None, num_images=1000, preproc_resize=(256, 256), log=False, eval_dir=None):
        if  option == "exit":
            exit_msg = {"cmd":option}
            self.send_data(exit_msg)
            click.echo("Exit message sent. Closing connection.")
            try:
                response = self.client_socket.recv(self.buffer_size).decode('utf-8')
                click.echo(response)
            except:
                click.echo("No response from server (likely already closed).")
            self.client_socket.close()
            return
        
        elif option == "mlist":
            params ={
                "cmd"       : option,
                "task"      : task,
                "eval_dir"  : eval_dir
            }
            self.send_data(params)
            result = self.receive_data()
            self.display_model_list(result)


        else:
            if not(model_name or lne_path):
                raise ValueError("One of model_name or lne_path must be provided")
            params = {
                "cmd"           : option,
                "task"          : task,
                "model_name"    : model_name,
                "lne_path"      : lne_path,
                "num_images"    : num_images,
                "preproc_resize": preproc_resize,
                "log": log,
                "eval_dir"  : eval_dir
            }

            # Send request
            self.send_data(params)
            click.echo("Model Evaluation request sent. Waiting for response...")

            # Receive result
            result = self.receive_data()
            if isinstance(result, str):
                click.echo(str)
            else:
                self.display_test_result(result)
            if log: 
                self.save_log_data(result.get('model_name', 'unknown_model'), result.get('log_data', {}))


    def display_model_list(self, result):
        click.echo("==================== ModelList====================")
        for category, info in result.items():
            click.echo(f"\n{category}: {info['dir']}")

            for model in info["models"]:
                click.echo(f"\t{model['name']}")


    def display_test_result(self, result):
        filtered_result = {}
        for k, v in result.items():
            if k == 'log_data':
                continue
            try:
                numeric_value = float(v)
                filtered_result[k] = round(numeric_value, 4)  # 소수점 4자리까지 반올림
            except (ValueError, TypeError):
                filtered_result[k] = v

        model_name  = filtered_result.get('model_name', 'Unknown Model')
        lne_path    = filtered_result.get('lne_path', 'N/A')
        avg_fps     = filtered_result.get('average_fps', 'N/A')
        min_fps     = filtered_result.get('min_fps', 'N/A')
        max_fps     = filtered_result.get('max_fps', 'N/A')
        top1_acc    = filtered_result.get('top1_accuracy', 'N/A')
        top5_acc    = filtered_result.get('top5_accuracy', 'N/A')

        click.echo(f"======== Image Classification for {model_name} ========")
        click.echo(f"  LNE file       : {lne_path}")
        click.echo(f"  Average FPS    : {avg_fps}")
        click.echo(f"      min FPS    : {min_fps}")
        click.echo(f"      max FPS    : {max_fps}")
        click.echo(f"  Top-1 Accuracy : {top1_acc}")
        click.echo(f"  Top-5 Accuracy : {top5_acc}")


    def save_log_data(self, model_name, log_data):
        log_path = f"output/{model_name}.log"
        with open(log_path, 'w') as json_file:
            json.dump(log_data, json_file, indent=4)
        click.echo(f"log_data saved to {log_path}")


if __name__ == "__main__":
    client = Client(server_ip="192.168.0.163")

    try:
        # 예시: classify 명령 보내기
        client.communicate(
            option="test",
            task="classify",
            model_name=None,
            num_images=100,
            preproc_resize=(256, 256),
            log=True
        )

        # 예시: exit 명령 보내기
        client.communicate(option="exit")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        client.client_socket.close()