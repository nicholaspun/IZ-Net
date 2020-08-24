from matplotlib import pyplot as plt
import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from consts import ALL_MEMBERS, FACES_PATH
from util import load_img

class Analysis:
    def __init__(self, model):
        print("Intializing Analysis class ... ")
        self.embeddings = self.__get_embeddings(model)
        print("Intialization Complete.")

    def __get_embeddings(self, model):
        embeddings = {}

        for member in ALL_MEMBERS:
            embeddings[member] = []
            for image_name in os.listdir(os.path.join(FACES_PATH, member)):
                embeddings[member].append(
                    np.squeeze(model(load_img(os.path.join(FACES_PATH, member, image_name)))))

        return embeddings
            
    def visualize_tsne(self, save_path, **kwargs):
        print("Doing some pre-analysis work ...")
        embeddings_list = []
        embeddings_ordering = []
        for member, member_embeddings in self.embeddings.items():
            embeddings_list += member_embeddings
            embeddings_ordering.append(member)

        # Fit embeddings
        print("Reducing dimensions of embeddings ...")
        embeddings_reduced = PCA(50).fit_transform(embeddings_list)
        print("Fitting embeddings ...")
        embeddings_fitted = TSNE(**kwargs).fit_transform(embeddings_reduced)

        # Visualize embeddings
        plt.figure(figsize=(10, 10))

        colors = ['#2f4f4f', "#8b4513", "#228b22", "#00008b", "#ff0000", "#ffd700",
                "#7fff00", "#00ffff", "#ff00ff", "#1e90ff", "#ffe4b5", "#ff69b4"]
        left = 0
        right = 0
        for index, member in enumerate(embeddings_ordering):
            left = right
            right = left + len(self.embeddings[member])
                
            x_coords = [coord[0] for coord in embeddings_fitted[left:right]]
            y_coords = [coord[1] for coord in embeddings_fitted[left:right]]

            plt.scatter(x_coords, y_coords, c=colors[index], label=member)

        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        plt.savefig(save_path)
        plt.show()

    def __pairwise_distances(self, embeddings):
        gram = np.matmul(embeddings, np.transpose(embeddings))
        norms = np.diag(gram)
        dist_mat = np.expand_dims(norms, 0) - 2.0 * gram + np.expand_dims(norms, 1)
        return list(dist_mat[np.triu_indices(len(embeddings), 1)])

    def visualize_pairwise_distance_boxplot(self, save_path):
        distances = []
        member_ordering = []
        for member, member_embeddings in self.embeddings.items():
            print("Calculating pairwise distance for {}".format(member))
            member_ordering.append(member)
            distances.append(self.__pairwise_distances(member_embeddings))
        
        plt.figure(figsize=(10, 10))
        plt.boxplot(distances)
        plt.ylabel("Pairwise Distance (Euclidean)")
        plt.xticks(np.arange(1, len(member_ordering) + 1), member_ordering)
        plt.savefig(save_path)
        plt.show()
