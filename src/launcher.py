from llamafactory.train.tuner import run_exp,export_model
from llamafactory.eval.evaluator import run_eval

def launch():
    
    run_exp()
    # export_model()


if __name__ == "__main__":
    launch()
