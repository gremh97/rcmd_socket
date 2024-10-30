import sys, os
import socket
import json
import click
import tabulate
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


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
        pbar = None
        end_marker = b"$END$" # termination marker for large data
        while True:
            try:
                chunk = self.client_socket.recv(self.buffer_size)
                if not chunk:
                    break  

                response += chunk

                current_data = response.decode('utf-8', errors='ignore')
                
                # marker for large data
                if end_marker in response:
                    complete_message = current_data.replace(end_marker.decode('utf-8'), '')
                    if pbar:
                        pbar.close()
                    try:
                        return json.loads(complete_message)
                    except json.JSONDecodeError:
                        return complete_message 

                # marker for small data '\n'
                if '\n' in current_data:
                    lines = current_data.split('\n')
                    for line in lines:
                        if not line:
                            continue
                        try:
                            message = json.loads(line)
                            
                            if message.get("type") == "progress":
                                if pbar is None:
                                    pbar = tqdm(total=message["total"], desc="Processing")
                                pbar.update(message["current"] - pbar.n)
                                response = b""
                            else:
                                if pbar:
                                    pbar.close()
                                    print('\n')
                                return message
                        except json.JSONDecodeError:
                            continue
            except socket.timeout:
                if pbar:
                    pbar.close()
                click.echo("Timeout while receiving data", err=True)
                break

        if pbar:
            pbar.close()
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
                click.echo(result)
            elif result.get("type") == "complete_classifier":
                self.display_classifier_result(result)
            else:
                self.display_yolo_result(result)


            if log: 
                self.save_log_data(result.get('model_name', 'unknown_model'), result.get('log_data', {}))


    def display_model_list(self, result):
        click.echo("====================== ModelList ======================")
        for category, info in result.items():
            click.echo(f"\n{category}: {info['dir']}")

            for model in info["models"]:
                click.echo(f"\t{model['name']}")
        click.echo("=======================================================")


    def display_classifier_result(self, result):
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

        title = f"\n\n================== Image Classification for {model_name} =================="
  
        click.echo(title)
        click.echo(f"  LNE file       : {lne_path}")
        click.echo(f"  Average FPS    : {avg_fps}")
        click.echo(f"      min FPS    : {min_fps}")
        click.echo(f"      max FPS    : {max_fps}")
        click.echo(f"  Top-1 Accuracy : {top1_acc}")
        click.echo(f"  Top-5 Accuracy : {top5_acc}")
        click.echo(f"{'='*len(title)}")


    def display_yolo_result(self, result):
        model_name = result.get("model_name")
        detections = result.get("detections")
        avg_fps    = result.get("average_fps")
        
        # Suppress output messages
        def suppress_stdout():
            sys.stdout = open(os.devnull, "w")

        def restore_stdout():
            sys.stdout.close()
            sys.stdout = sys.__stdout__
        
       # Save detections to a JSON file in COCO format
        with open("detections.json", "w") as f:
            json.dump(detections, f)

        suppress_stdout()
        # Load COCO ground truth and detections
        coco_gt = COCO("/data/image/coco/annotations/instances_val2017.json")   # Provide the path to annotations.json
        coco_dt = coco_gt.loadRes("detections.json")
        restore_stdout()

        # Initialize and run COCO evaluation
        # Redirect stdout to suppress print statements
        suppress_stdout()
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        restore_stdout()
            
        # Extract AP50 and mAP for each category
        categories = coco_gt.loadCats(coco_gt.getCatIds())
        category_names = [cat["name"] for cat in categories]
        category_AP50 = []
        category_mAP = []

        for i, cat in enumerate(category_names):
            # Precision for the first IoU threshold (0.5) across all categories
            precision_AP50 = coco_eval.eval["precision"][0, :, i, 0, -1]
            precision_AP50 = precision_AP50[precision_AP50 > -1]
            AP50 = np.mean(precision_AP50) if precision_AP50.size else float("nan")
            category_AP50.append(float(AP50 * 100))

            # Precision for IoU range (0.5 to 0.95)
            precision_mAP = coco_eval.eval["precision"][:, :, i, 0, -1]
            precision_mAP = precision_mAP[precision_mAP > -1]
            mAP = np.mean(precision_mAP) if precision_mAP.size else float("nan")
            category_mAP.append(float(mAP * 100))

        # Print class-wise AP50 and mAP
        table_data = np.stack([category_names, category_AP50, category_mAP]).T
        headers = ["Class", "AP50", "mAP"]

        title = f"\n========================== Image Detection for {model_name} ===========================\n"
        click.echo(title)
        coco_eval.summarize()
        click.echo("\nClass-wise AP50 and mAP:")
        click.echo(tabulate.tabulate(table_data, headers, tablefmt="pipe", floatfmt=".1f"))
        click.echo(f"\nALL CLASS Average AP50     : {sum(category_AP50)/len(category_AP50):.3f}")
        click.echo(f"ALL CLASS Average mAP50    : {sum(category_mAP)/len(category_mAP):.3f}")
        click.echo(f"   Average FPS             : {avg_fps:.3f}")
        click.echo(f"{'='*len(title)}\n")
        
        if os.path.exists("detections.json"):
            os.remove("detections.json")


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