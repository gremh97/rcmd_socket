import os
import argparse
from server import Server
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument('-p','--port', type=int, required=False, default=5000,  help="Port number to use")
args                = parser.parse_args()
port                = args.port
eval_dir            = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
model_json_path     = os.path.join(os.path.dirname(__file__), 'models_by_task.json')

server              = Server(port=port, model_json_path=model_json_path, eval_dir=eval_dir)
server.start()
