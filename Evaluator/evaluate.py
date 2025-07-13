import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from Inference_utilities.evaluator import SceneTextEvaluator
from Inference_utilities.tee_logger import Tee

# Save print outputs to a file
os.makedirs("eval_out",exist_ok=True)
sys.stdout = Tee("eval_out/evaluation_log.txt")

evaluator = SceneTextEvaluator("Configs/config.yaml")
precision = evaluator.evaluate()
