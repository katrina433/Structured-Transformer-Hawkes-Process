import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('_mpl-gallery')
import numpy as np
import glob, os
import seaborn as sns
import pickle as pkl

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
                
    # select top 3-5 most likely options (according to log likelihood) for ease of plotting
    top_n = 5
    # sorts in ascending order
    indices = np.argsort(np.array(metrics_data[metrics[0]]))[-top_n:]
    selected_filenames = np.array(file_names)[indices]
    selected_ll = np.array(metrics_data[metrics[0]])[indices]
    selected_acc = np.array(metrics_data[metrics[1]])[indices]
    selected_rmse = np.array(metrics_data[metrics[2]])[indices]
    
    print(selected_filenames)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    for ax in (ax1, ax2, ax3):
        ax.set_xticks([])
    fig.suptitle("mean validation metrics")
    ax1.bar(range(top_n), selected_ll, label=selected_filenames, color=cmap()(np.linspace(0.0, 0.9, top_n)))
    ax1.set(ylim=(-0.8, 0.2))
    ax1.set_title("log-likelihood")
    ax2.bar(range(top_n), selected_acc, label=selected_filenames, color=cmap()(np.linspace(0.0, 0.9, top_n)))
    ax2.set(ylim=(0.5, 0.9))
    ax2.set_title("accuracy")
    ax3.bar(range(top_n), selected_rmse, label=selected_filenames, color=cmap()(np.linspace(0.0, 0.9, top_n)))
    ax3.set(ylim=(1.0, 1.3))
    ax3.set_title("RMSE")
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    fig.set_size_inches(10, 5)
    fig.tight_layout()
    fig.savefig("comparison.png", dpi = 100)
# bar()

def plotA():
    # 75 x 75 matrix representing node (zipcode areas) interactions on 911 calls
    A=np.load("911_network_structure/A_cumulative_dt_max_20.npy")
    W=np.load("911_network_structure/W_cumulative_dt_max_20.npy")
    # Info we might want to collect:
    # some measure of sparsity, like average degree, or a heatmap
    # size of connected components?
    print(A*W)
    # TODO: set x and y axis to zipcode number?
    fr = open("data/zip_mapping.pkl", "rb")
    node_to_zip = pkl.load(fr)
    print(node_to_zip)
    zips = []
    for i in range(A.shape[0]):
        zips.append(node_to_zip[i+1])
    sns.heatmap(data=(A*W), xticklabels=zips, yticklabels=zips, annot=False)
    fig = plt.gcf()
    fig.set_size_inches(6, 6)
    plt.title("Interaction between zipcodes")
    plt.tick_params(axis='both', which='major', labelsize=5)
    plt.ylabel("Initiating node")
    plt.xlabel("Receiving node")
    plt.tight_layout()
    fig.savefig("implic_net_struct.png", dpi=150)
plotA()