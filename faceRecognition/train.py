from keras import backend as K
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, Input, Layer, Lambda

from util import get_pairs, get_triplets

# Taken from https://github.com/CrimyTheBold/tripletloss/blob/master/02%20-%20tripletloss%20MNIST.ipynb
class TripletLossLayer(Layer):
    def __init__(self, alpha=0.2, **kwargs):
        self.alpha = alpha
        super(TripletLossLayer, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        anchor, positive, negative = inputs
        p_dist = K.sum(K.square(anchor - positive), axis=-1)
        n_dist = K.sum(K.square(anchor - negative), axis=-1)
        return K.sum(K.maximum(p_dist - n_dist + self.alpha, 0), axis=0)

    def call(self, inputs):
        loss = self.triplet_loss(inputs)
        self.add_loss(loss)
        return loss

def train_with_triplet_loss(shared_network, num_iterations, num_triplets, save_path):
    anchor = Input((224, 224, 3))
    positive = Input((224, 224, 3))
    negative = Input((224, 224, 3))

    anchor_encoding = shared_network(anchor)
    positive_encoding = shared_network(positive)
    negative_encoding = shared_network(negative)

    loss_layer = TripletLossLayer(alpha=0.2, name='triplet_loss_layer')(
        [anchor_encoding, positive_encoding, negative_encoding])

    model = Model(inputs=[anchor, positive, negative], outputs=loss_layer)

    opt = Adam(clipnorm=1.0, clipvalue=0.5)
    model.compile(loss=None, optimizer=opt)

    training_losses = []
    for i in range(1, num_iterations + 1):
        print('Start iteration {} of {}'.format(i, num_iterations))
        triplets = get_triplets(num_triplets, shared_network)
        print('Triplets acquired')
        train_loss = model.train_on_batch(triplets, None)
        print('Training Loss: {}'.format(train_loss))
        training_losses.append(train_loss)

        if i % 20 == 0:
            shared_network.save_weights(save_path)

    return training_losses

def train_as_siamese_network(shared_network, num_iterations, samples_per_iteration, save_path):
    sample_1 = Input((224, 224, 3))
    sample_2 = Input((224, 224, 3))

    sample_1_encoding = shared_network(sample_1)
    sample_2_encoding = shared_network(sample_2)

    output = Lambda(lambda pair: K.abs(
        pair[0] - pair[1]))([sample_1_encoding, sample_2_encoding])
    output = Dense(1, activation='sigmoid')(output)

    model = Model(inputs=[sample_1, sample_2], outputs=output)

    opt = Adam(clipnorm=1.0, clipvalue=0.5)
    model.compile(loss="binary_crossentropy", optimizer=opt,
                  metrics=['binary_accuracy'])

    training_losses = []
    for i in range(1, num_iterations + 1):
        print('Start iteration {} of {}'.format(i, num_iterations))
        left, right, expected = get_pairs(samples_per_iteration)
        print('Samples acquired')
        train_loss = model.train_on_batch([left, right], expected)
        print('Training Loss: {}'.format(train_loss))
        training_losses.append(train_loss)

        if i % 20 == 0:
            shared_network.save_weights(save_path)

    return training_losses
