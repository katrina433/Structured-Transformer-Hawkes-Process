import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('_mpl-gallery')
import numpy as np
import glob, os

def cmap(name='tab20'):
    return plt.get_cmap(name)

# performance comparison
def bar():
    metrics = ["ll", "accuracy", "RMSE"]
    metrics_data = {"ll" : [], "accuracy" : [], "RMSE": []}
    
    # read files from 911_results
    files = glob.glob("911_results/*.txt")
    file_names = [os.path.basename(file)[4:-4] for file in files] # groups for plotting
    print(file_names)
    for file in files:
        with open(file) as fr:
            # read last line and add to list
            lines = fr.readlines()
            metrics_txt = lines[len(lines)-1]
            # print(metrics_txt)
            match = [metrics_txt[20:28], metrics_txt[57:64], metrics_txt[89:96]]
            # print(match)
            for metric, idx in zip(metrics, range(3)):
                metrics_data[metric].append(float(match[idx]))
                
    # may want to only select top 3-5 options for ease of plotting
    
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    for ax in (ax1, ax2, ax3):
        ax.set_xticks([])
    fig.suptitle("mean validation metrics")
    ax1.bar(range(len(file_names)), np.array(metrics_data[metrics[0]]), label=file_names, color=cmap()(np.linspace(0.0, 0.9, len(file_names))))
    ax1.set(ylim=(-0.5, 0.5))
    ax1.set_title("log-likelihood")
    # ax1.legend()
    ax2.bar(range(len(file_names)), metrics_data[metrics[1]], label=file_names, color=cmap()(np.linspace(0.0, 0.9, len(file_names))))
    ax2.set(ylim=(0, 1))
    ax2.set_title("accuracy")
    ax3.bar(range(len(file_names)), metrics_data[metrics[2]], label=file_names, color=cmap()(np.linspace(0.0, 0.9, len(file_names))))
    ax3.set_title("RMSE")
    fig.set_size_inches(8, 3)
    fig.tight_layout()
    fig.savefig("comparison.png", dpi = 100)
bar()