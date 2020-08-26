from matplotlib import pyplot as plt
import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from consts import ALL_MEMBERS, FACES_PATH
import util

class Analysis:
    def __init__(self, model):
        print("Intializing Analysis class ... ")
        self.embeddings = self.__get_embeddings(model)
        self.medoids_index = {}
        print("Intialization Complete.")

    def __get_embeddings(self, model):
        embeddings = {}

        for member in ALL_MEMBERS:
            print('Calculating embeddings for {} ...'.format(member))
            embeddings[member] = []
            for image_name in os.listdir(os.path.join(FACES_PATH, member)):
                embeddings[member].append(
                    np.squeeze(model(util.load_img(os.path.join(FACES_PATH, member, image_name)))))

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
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def visualize_pairwise_distance_boxplot(self, save_path):
        distances = []
        member_ordering = []
        for member, member_embeddings in self.embeddings.items():
            print("Calculating pairwise distance for {}".format(member))
            member_ordering.append(member)
            pairwise_distance_matrix = util.compute_pairwise_distance_matrix(member_embeddings)
            distances.append(
                list(pairwise_distance_matrix[np.triu_indices(len(member_embeddings), 1)]))

            if member not in self.medoids_index:
                medoid_embedding_index = np.argmin(np.sum(pairwise_distance_matrix, axis=1))
                self.medoids_index[member] = medoid_embedding_index
        
        plt.figure(figsize=(10, 10))
        plt.boxplot(distances)
        plt.ylabel("Pairwise Distance (Euclidean)")
        plt.xticks(np.arange(1, len(member_ordering) + 1), member_ordering)
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()

    def visualize_medoids_table(self, save_path):
        for member in ALL_MEMBERS:
            if member not in self.medoids_index:
                pairwise_distance_matrix = util.compute_pairwise_distance_matrix(self.embeddings[member])
                medoid_embedding_index = np.argmin(np.sum(pairwise_distance_matrix, axis=1))
                self.medoids_index[member] = medoid_embedding_index
        
        member_order = []
        medoid_matrix = []
        for member, member_medoid_embedding_index in self.medoids_index.items():
            member_order.append(member)
            medoid_matrix.append(
                self.embeddings[member][member_medoid_embedding_index])
        
        medoid_pairwise_distance_mat = util.compute_pairwise_distance_matrix(medoid_matrix)

        plt.figure(figsize=(20, 5))

        ax = plt.gca()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.box(on=None)

        the_table = plt.table(
            cellText=medoid_pairwise_distance_mat,
            rowLabels=member_order,
            colLabels=member_order,
            loc='center',
            rowLoc='right',
            colLoc='center'
        )
        the_table.scale(1, 1.5)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight')
        plt.show()

