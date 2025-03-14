from tensorboard import program
# import matplotlib.pyplot as plt


def plot_training_metrics(log_dir="./logs"):
    # Plot loss curves
    # Plot evaluation metrics
    pass


def monitor_training():
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', './logs'])
    url = tb.launch()
    print(f"TensorBoard running at {url}")
