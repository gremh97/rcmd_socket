import socket
import os
import json
import time
from threading import Thread, Lock

from ModelEvaluator import ClassifierEvaluator
from json_utils import update_model_list, find_task_by_model


class Server:
    def __init__(self, host=None, port=5000, buffer_size=4096, server_timeout=10, model_json_path=None, eval_dir=None):
        self.host = host or Server.get_IP_addr()
        self.port = port
        self.buffer_size = buffer_size
        self.server_timeout = server_timeout
        self.is_active = True
        self.clients = []
        self.lock = Lock()  # Lock for thread-safe client list operations
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            print(f"Server listening on {self.host}:{self.port}")
        except socket.error as err:
            print(f"Server socket creation failed with error {err}")
        
        self.model_json_path = model_json_path
        self.eval_dir = eval_dir

    def start(self):
        print("Server is listening for incoming connections...")
        try:
            while self.is_active:
                try:
                    # 10초 동안 클라이언트 대기
                    self.server_socket.settimeout(self.server_timeout)
                    client_socket, client_address = self.server_socket.accept()
                    with self.lock:
                        self.clients.append(client_socket)
                    client_handler = Thread(target=self.handle_client, args=(client_socket, client_address))
                    client_handler.start()
                except socket.timeout:
                    print(f"No connection requests within the last {self.server_timeout} seconds.")
                    with self.lock:
                        if not self.clients:
                            print("No more connected clients. Shutting down the server.")
                            self.is_active = False
                        else:
                            print(f"Clients are still connected. Continuing to listen.")
        finally:
            self.close_all_clients()
            self.server_socket.close()
            print("Server socket closed.")

    def handle_client(self, client_socket, client_address):
        print(f"Connected to client({client_address}:{self.port})")
        try:
            while True:
                data = client_socket.recv(self.buffer_size).decode('utf-8')
                if not data:
                    break

                params = json.loads(data)
                cmd = params.get("cmd")

                if cmd == "exit":
                    print("Exit task received. Closing client connection.")
                    client_socket.sendall(b"Server received exit task. Closing your connection.")
                    break

                elif cmd == "mlist":
                    print("...Processing model list...")
                    task = params.get("task") or ['classify', 'detect', 'detect_yolo']
                    self.eval_dir = params.get('eval_dir') or self.eval_dir
                    model_list = update_model_list(task, self.eval_dir, self.model_json_path)
                    json_result = json.dumps(model_list)

                else:
                    print("...Reading params from client... ")
                    model_name = params['model_name']
                    print(model_name)
                    task = params.get('task') or find_task_by_model(model_name, self.model_json_path)
                    if not task:
                        client_socket.sendall(f"The model does not exist: {model_name}".encode('utf-8'))
                        continue
                    self.eval_dir = params.get('eval_dir') or self.eval_dir
                    lne_path = params['lne_path'] or os.path.join(self.eval_dir, task, 'input', f"{model_name}.lne")
                    num_images = params['num_images']
                    preproc_resize = tuple(params.get('preproc_resize', (256, 256)))
                    log = params['log']

                    print("...Checking the Performance of model...")
                    if task == "classify":
                        evaluator = ClassifierEvaluator(
                            model_name=model_name,
                            lne_path=lne_path,
                            num_images=num_images,
                            preproc_resize=preproc_resize,
                            log=log
                        )
                    elif task == "detect":
                        pass
                    else:
                        client_socket.sendall(b"Invalid task")
                        continue
                    json_result = json.dumps(evaluator.evaluate())

                client_socket.sendall(json_result.encode('utf-8'))

        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            with self.lock:
                self.clients.remove(client_socket)  # 클라이언트 소켓 제거
            client_socket.close()
            print(f"Connection closed: {client_address}")

    def close_all_clients(self):
        print("Closing all client connections...")
        with self.lock:
            for client_socket in self.clients:
                try:
                    client_socket.close()
                    print("Closed client socket.")
                except Exception as e:
                    print(f"Error closing client socket: {e}")
            self.clients = []

    @staticmethod
    def get_IP_addr():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip

if __name__ == "__main__":
    server = Server()
    server.start()
