from keras import backend as K
from keras.models import Model
from keras.layers import BatchNormalization, Conv2D, Dense, Flatten, Input, Lambda, MaxPool2D

def base_network():
  X_input = Input((224, 224, 3))

  X = Conv2D(64, (7, 7), strides=(2, 2), activation='relu',
             padding='same', data_format='channels_last')(X_input)
  X = BatchNormalization()(X)
  X = MaxPool2D(strides=(1, 1))(X)

  X = Conv2D(64, (1, 1), activation='relu')(X)
  X = Conv2D(192, (3, 3), activation='relu', padding='same')(X)
  X = BatchNormalization()(X)
  X = MaxPool2D(padding='same')(X)

  X = Conv2D(192, (1, 1), activation='relu')(X)
  X = Conv2D(384, (3, 3), activation='relu', padding='same')(X)
  X = BatchNormalization()(X)
  X = MaxPool2D()(X)

  X = Conv2D(384, (1, 1), activation='relu')(X)
  X = Conv2D(256, (3, 3), activation='relu', padding='same')(X)
  X = Conv2D(256, (1, 1), activation='relu')(X)
  X = Conv2D(256, (3, 3), activation='relu', padding='same')(X)
  X = Conv2D(256, (1, 1), activation='relu')(X)
  X = Conv2D(256, (3, 3), activation='relu', padding='same')(X)
  X = BatchNormalization()(X)
  X = MaxPool2D()(X)

  X = Flatten()(X)
  X = Dense(4096, activation='relu')(X)
  X = Dense(4096, activation='relu')(X)

  # X_d1 = Dense(4096)(X)
  # X_d2 = Dense(4096)(X)
  # X = Maximum()([X_d1, X_d2])

  # X_d1 = Dense(4096)(X)
  # X_d2 = Dense(4096)(X)
  # X = Maximum()([X_d1, X_d2])

  X = Dense(128, activation='relu')(X)
  X = Lambda(lambda x: K.l2_normalize(x, axis=1))(X)

  return Model(inputs=X_input, outputs=X)
