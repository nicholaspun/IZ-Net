import cv2
import numpy as np
import os
import random

from consts import ALL_MEMBERS, FACES_PATH

def load_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img[..., ::-1]  # Reverse channels
    img = np.around(img/255.0, decimals=12)  # Normalize
    return np.array([img])

def compute_pairwise_distance_matrix(embeddings):
    gram = np.matmul(embeddings, np.transpose(embeddings))
    norms = np.diag(gram)
    dist_mat = np.expand_dims(norms, 0) - 2.0 * gram + np.expand_dims(norms, 1)
    return dist_mat

def compute_pairwise_distance_matrix_between_2_sets_of_embeddings(embeddings_1, embeddings_2):
    norms_1 = np.diag(np.matmul(embeddings_1, np.transpose(embeddings_1)))
    norms_2 = np.diag(np.matmul(embeddings_2, np.transpose(embeddings_2)))
    dot_product_mat = np.matmul(embeddings_1, np.transpose(embeddings_2))
    return np.expand_dims(norms_1, 1) - 2 * dot_product_mat + np.expand_dims(norms_2, 0)

def get_triplets(num, network, alpha=0.2):
    anchor = np.empty((0, 224, 224, 3))
    positive = np.empty((0, 224, 224, 3))
    negative = np.empty((0, 224, 224, 3))
    original_num = num

    while num > 0:
        member1, member2 = random.sample(ALL_MEMBERS, k=2)

        pos_img_names = os.listdir(os.path.join(FACES_PATH, member1))
        random.shuffle(pos_img_names) # eck, mutation
        pos_img_names = pos_img_names[:original_num] # So the computation isn't as heavy

        neg_img_names = os.listdir(os.path.join(FACES_PATH, member2))
        random.shuffle(neg_img_names)
        neg_img_names = neg_img_names[:original_num]

        embeddings_pos = [np.squeeze(network(load_img(os.path.join(FACES_PATH, member1, img_name)))) for img_name in pos_img_names]
        embeddings_neg = [np.squeeze(network(load_img(os.path.join(FACES_PATH, member2, img_name)))) for img_name in neg_img_names]

        pairwise_dist_pos = compute_pairwise_distance_matrix(embeddings_pos)
        pairwise_dist_neg = compute_pairwise_distance_matrix_between_2_sets_of_embeddings(embeddings_pos, embeddings_neg)

        triplet_dists = np.expand_dims(pairwise_dist_pos, 2) - np.expand_dims(pairwise_dist_neg, 1) + alpha

        index_anch, index_pos, index_neg = np.nonzero(triplet_dists > 0)

        for ia, ip, ine in zip(index_anch, index_pos, index_neg):
            if ia == ip:
                continue

            anchor = np.append(anchor, load_img(os.path.join(FACES_PATH, member1, pos_img_names[ia])), axis=0)
            positive = np.append(positive, load_img(os.path.join(FACES_PATH, member1, pos_img_names[ip])), axis=0)
            negative = np.append(negative, load_img(os.path.join(FACES_PATH, member2, neg_img_names[ine])), axis=0)
            num -= 1

            if num == 0:
                return [np.array(anchor), np.array(positive), np.array(negative)]
        
        # ----------------------------------
        # Previous, randomized approach
        # ----------------------------------
        # anchor_path, pos_path = random.sample(os.listdir(os.path.join(FACES_PATH, member1)), k=2)
        # neg_path = random.sample(os.listdir(os.path.join(FACES_PATH, member2)), k=1)[0]

        # candidate_anchor = load_img(os.path.join(FACES_PATH, member1, anchor_path))
        # candidate_positive = load_img(os.path.join(FACES_PATH, member1, pos_path))
        # candidate_negative = load_img(os.path.join(FACES_PATH, member2, neg_path))

        # embedding_anchor = network(candidate_anchor)
        # embedding_positive = network(candidate_positive)
        # embedding_negative = network(candidate_negative)

        # if (np.linalg.norm(embedding_anchor - embedding_positive) + 0.2 >= np.linalg.norm(embedding_anchor - embedding_negative)):
        #     anchor = np.append(anchor, candidate_anchor, axis=0)
        #     positive = np.append(positive, candidate_positive, axis=0)
        #     negative = np.append(negative, candidate_negative, axis=0)
        #     num -= 1

def get_pairs(num_samples):
    left = np.empty((0, 224, 224, 3))
    right = np.empty((0, 224, 224, 3))
    out = np.zeros((0, 1))

    for _ in range(num_samples // 2):
        member = random.sample(os.listdir(FACES_PATH), k=1)[0]
        sample_left_path, sample_right_path = random.sample(
            os.listdir(os.path.join(FACES_PATH, member)), k=2)
        left_img = load_img(os.path.join(
            FACES_PATH, member, sample_left_path))
        right_img = load_img(os.path.join(
            FACES_PATH, member, sample_right_path))

        left = np.append(left, left_img, axis=0)
        right = np.append(right, right_img, axis=0)
        out = np.append(out, [[1]], axis=0)

    for _ in range(num_samples // 2):
        member_left, member_right = random.sample(
            os.listdir(FACES_PATH), k=2)
        left_img = load_img(
            os.path.join(FACES_PATH, member_left, random.sample(os.listdir(os.path.join(FACES_PATH, member_left)), k=1)[0]))
        right_img = load_img(
            os.path.join(FACES_PATH, member_right, random.sample(os.listdir(os.path.join(FACES_PATH, member_right)), k=1)[0]))

        left = np.append(left, left_img, axis=0)
        right = np.append(right, right_img, axis=0)
        out = np.append(out, [[0]], axis=0)

    return [left, right, out]



