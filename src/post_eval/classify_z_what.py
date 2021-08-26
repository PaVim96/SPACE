import torch
import torch.nn.functional as F
from simple_net import Network
from sklearn.linear_model import LogisticRegression, LinearRegression, SGDClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import warnings
from sklearn import metrics
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from argparse import Namespace
import os
from sklearn.neighbors import KNeighborsClassifier
from dataset import get_label_list
from collections import Counter

N_NEIGHBORS = 24
DISPLAY_CENTROIDS = True

warnings.filterwarnings("ignore", category=UserWarning)
import argparse
import numpy as np

base_objects_colors = {"sue": (180, 122, 48), "inky": (84, 184, 153),
                       "pinky": (198, 89, 179), "blinky": (200, 72, 72),
                       "pacman": (210, 164, 74), "fruit": (184, 50, 50),
                       "save_fruit": (144, 10, 10),
                       "white_ghost": (214, 214, 214),
                       "blue_ghost": (66, 114, 194), "score0": (200, 50, 200),
                       "life": (70, 210, 70), "life2": (20, 150, 20), "no_label": (160, 160, 160)
                       }


def evaluate_z_what(arguments, z_what, labels, n, cfg, title="moving"):
    """
    :param arguments: dict of properties
    :param z_what: (#objects, encoding_dim)
    :param labels: (#objects)
    # :param z_where: (#objects, 4)
    :param cfg:
    :return:
        result: metrics
        path: to pca
        accuracy: few shot accuracy
    """

    c = Counter(labels.tolist())
    print("Distribution of matched labels:", c)
    relevant_labels = [int(part) for part in arguments['indices'].split(',')] if arguments['indices'] else list(c.keys())
    folder = f'hyper/{cfg.exp_name}{cfg.seed}' if cfg else f'{arguments["folder"]}1'
    pca_path = f"../output/logs/{folder}/pca{arguments['indices'] if arguments['indices'] else ''}_{title}.png"
    label_list = get_label_list(cfg)

    relevant = torch.zeros(labels.shape, dtype=torch.bool)
    for rl in relevant_labels:
        relevant |= labels == rl
    z_what = z_what[relevant]
    labels = labels[relevant]
    train_portion = 0.9
    nb_sample = int(train_portion * len(labels))
    test_x = z_what[nb_sample:]
    test_y = labels[nb_sample:]
    train_x = z_what[:nb_sample]
    train_y = labels[:nb_sample]
    few_shot_accuracy = {}
    z_what_by_game = {rl: train_x[train_y == rl] for rl in relevant_labels}
    labels_by_game = {rl: train_y[train_y == rl] for rl in relevant_labels}
    for training_objects_per_class in [1, 4, 16, 64]:
        current_train_sample = torch.cat([z_what_by_game[rl][:training_objects_per_class] for rl in relevant_labels])
        current__train_labels = torch.cat([labels_by_game[rl][:training_objects_per_class] for rl in relevant_labels])
        clf = LogisticRegression()
        clf.fit(current_train_sample, current__train_labels)
        acc = clf.score(test_x, test_y)
        few_shot_accuracy[f'few_shot_accuracy_with_{training_objects_per_class}'] = acc

    clf = KMeans(n_clusters=len(relevant_labels))
    y = clf.fit_predict(z_what)
    results = {
        'adjusted_mutual_info_score': metrics.adjusted_mutual_info_score(labels, y),
        'adjusted_rand_score': metrics.adjusted_rand_score(labels, y),
    }
    centroids = clf.cluster_centers_
    X = train_x.numpy()
    nn = NearestNeighbors(n_neighbors=N_NEIGHBORS).fit(X)
    _, z_w_idx = nn.kneighbors(centroids)
    centroid_label = []
    for cent, nei in zip(centroids, z_w_idx):
        count = {rl: 0 for rl in relevant_labels}
        added = False
        for i in range(N_NEIGHBORS):
            nei_label = labels[nei[i]].item()
            count[nei_label] += 1
            if count[nei_label] > 6.0 / (i + 1) if nei_label in centroid_label else 3.0 / (i + 1):
                centroid_label.append(nei_label)
                added = True
                break
        if not added:
            leftover_labels = [i for i in relevant_labels if i not in centroid_label]
            centroid_label.append(leftover_labels[0])
    nn_class = KNeighborsClassifier(n_neighbors=1)
    nn_class.fit(centroids, centroid_label)
    few_shot_accuracy[f'few_shot_accuracy_cluster_nn'] = nn_class.score(test_x, test_y)

    train_all = torch.cat((z_what, labels.unsqueeze(1)), 1)
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'pink', 'purple', 'orange',
              'olive', 'brown', 'tomato', 'darkviolet', 'grey', 'chocolate']
    # SORT THE INDICES
    sorted = []
    for i in relevant_labels:
        mask = train_all.T[-1] == i
        indices = torch.nonzero(mask)
        sorted.append(indices)
    pca = PCA(n_components=arguments['dim'])
    z_what_emb = pca.fit_transform(z_what.numpy())
    if arguments['edgecolors']:
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(20, 12)
        ax.set_facecolor((0.3, 0.3, 0.3))
        plt.suptitle("Labeled PCA of z_whats", y=0.96, fontsize=28)
        plt.title("Inner Color is GT, Outer is greedy Centroid-based label", fontsize=18, pad=20)
        n = min(n, 10000)
        for i, idx in enumerate(sorted):
            if torch.numel(idx) == 0:
                continue
            y_idx = y[idx] if torch.numel(idx) > 1 else [[y[idx]]]
            if "pacman" in label_list:
                colr = [np.array(base_objects_colors[label_list[relevant_labels[i]]]) / 255]
                try:
                    edge_colors = [np.array(base_objects_colors[label_list[centroid_label[assign[0]]]]) / 255 for assign
                                   in
                                   y_idx]
                except:
                    print("=======================ERROR:", idx, y[idx], y)

            else:
                colr = colors[relevant_labels[i]]
                edge_colors = [colors[centroid_label[assign[0]]] for assign in y[idx]]
            ax.scatter(z_what_emb[:, 0][idx].squeeze()[:1],
                       z_what_emb[:, 1][idx].squeeze()[:1],
                       c=colr, label=label_list[relevant_labels[i]].replace("_", " "),
                       alpha=0.7)
            ax.scatter(z_what_emb[:, 0][idx].squeeze()[:n],
                       z_what_emb[:, 1][idx].squeeze()[:n],
                       c=colr,
                       alpha=0.7, edgecolors=edge_colors, s=100, linewidths=2)
        centroid_emb = pca.transform(centroids)

        if DISPLAY_CENTROIDS:
            for c_emb, cl in zip(centroid_emb, centroid_label):
                if "pacman" in label_list:
                    colr = [np.array(base_objects_colors[label_list[cl]]) / 255]
                else:
                    colr = colors[cl]
                ax.scatter([c_emb[0]],
                               [c_emb[1]],
                               c=colr,
                               edgecolors='black', s=100, linewidths=2)
        plt.legend(prop={'size': 6})
        directory = f"../output/logs/{folder}"
        if not os.path.exists(directory):
            print(f"Writing PCA to {directory}")
            os.makedirs(directory)
        plt.savefig(pca_path)
    else:
        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(30, 12)
        axs[0].set_facecolor((0.3, 0.3, 0.3))
        axs[1].set_facecolor((0.3, 0.3, 0.3))
        axs[0].set_title("Ground Truth Labels")
        axs[1].set_title("Labels Following Clustering")
        for i, idx in enumerate(sorted):
            # dimension issue only if there is exactly one object of one kind
            if torch.numel(idx) == 0:
                continue
            y_idx = y[idx] if torch.numel(idx) > 1 else [[y[idx]]]
            if "pacman" in label_list:
                colr = [np.array(base_objects_colors[label_list[relevant_labels[i]]]) / 255]
                try:
                    edge_colors = [np.array(base_objects_colors[label_list[centroid_label[assign[0]]]]) / 255 for assign
                                   in y_idx]
                except:
                    print("=======================ERROR:", idx, y[idx], y, y_idx)
                    raise
            else:
                colr = colors[relevant_labels[i]]
                edge_colors = [colors[centroid_label[assign[0]]] for assign in y_idx]
            axs[0].scatter(z_what_emb[:, 0][idx].squeeze(),
                           z_what_emb[:, 1][idx].squeeze(),
                           c=colr,
                           label=label_list[relevant_labels[i]],
                           alpha=0.7)
            axs[1].scatter(z_what_emb[:, 0][idx].squeeze(),
                           z_what_emb[:, 1][idx].squeeze(),
                           c=edge_colors,
                           label=label_list[relevant_labels[i]],
                           alpha=0.7)
        centroid_emb = pca.transform(centroids)

        for c_emb, cl in zip(centroid_emb, centroid_label):
            if "pacman" in label_list:
                colr = [np.array(base_objects_colors[label_list[cl]]) / 255]
            else:
                colr = colors[cl]
            axs[0].scatter([c_emb[0]],
                           [c_emb[1]],
                           c=colr,
                           edgecolors='black', s=100, linewidths=2)
            axs[1].scatter([c_emb[0]],
                           [c_emb[1]],
                           c=colr,
                           edgecolors='black', s=100, linewidths=2)

        axs[0].legend(prop={'size': 6})
        axs[1].legend(prop={'size': 6})
        if not os.path.exists(f"../output/logs/{folder}"):
            os.makedirs(f"../output/logs/{folder}")
        plt.tight_layout()
        plt.savefig(pca_path)
        plt.close(fig)
    return results, pca_path, few_shot_accuracy


# all_train_labels = pd.read_csv(f"../aiml_atari_data/rgb/MsPacman-v0/train_labels.csv")
# all_validation_labels = pd.read_csv(f"../aiml_atari_data/rgb/MsPacman-v0/validation_labels.csv")

# label_list = ["pacman", 'sue', 'inky', 'pinky', 'blinky', "blue_ghost",
#               "white_ghost", "fruit", "save_fruit", "life", "life2", "score0"]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the z_what encoding')
    parser.add_argument('-folder', type=str, default="mspacman_atari_example",
                        help='the output folder')
    parser.add_argument('-indices', type=str, default=None,
                        help='The relevant objects by their index, e.g. \"0,1\" for Pacman and Sue')
    parser.add_argument('-edgecolors', type=bool, default=True,
                        help='True iff the ground truth labels and the predicted labels '
                             '(Mixture of some greedy policy and NN) should be drawn in the same image')
    parser.add_argument('-dim', type=int, choices=[2, 3], default=2,
                        help='Number of dimension for PCA/TSNE visualization')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    nb_used_sample = 500

    # z_what_train = torch.randn((400, 32))
    # train_labels = torch.randint(high=8, size=(400,))
    z_what_train = torch.cat(torch.load(f"labeled/{args.folder}/z_what_validation.pt"))
    train_labels = torch.cat(torch.load(f"labeled/{args.folder}/labels_validation.pt"))

    evaluate_z_what(vars(args), z_what_train, train_labels, nb_used_sample, cfg=None)
