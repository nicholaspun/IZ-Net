from keras import backend as K
from keras.models import Model
from keras.layers import ZeroPadding2D, BatchNormalization, Concatenate, Conv2D, Dense, Input, MaxPool2D, Flatten, Lambda, AveragePooling2D

def l2_norm2d(x, pool_size=(2, 2), strides=None, padding='valid', data_format=None):
    if strides is None:
        strides = pool_size

    x = x ** 2
    output = K.pool2d(x, pool_size, strides, padding,
                      data_format, pool_mode='avg')
    output = K.sqrt(output)

    return output

def block_2(X):
  X = Conv2D(64, (1, 1), activation='relu')(X)
  X = Conv2D(192, (3, 3), padding='same', activation='relu')(X)
  X = BatchNormalization()(X)

  X = Conv2D(64, (1, 1), activation='relu')(X)
  X = Conv2D(192, (3, 3), padding='same', activation='relu')(X)
  X = BatchNormalization()(X)

  X = MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(X)

  return X

def block_3ab(X, num_filters, maxpool=True):
  X_1x1 = Conv2D(num_filters[0], (1, 1), activation='relu')(X)
  X_1x1 = BatchNormalization()(X_1x1)

  X_3x3 = Conv2D(num_filters[1], (1, 1), activation='relu')(X)
  X_3x3 = Conv2D(num_filters[2], (3, 3),
                 activation='relu', padding='same')(X_3x3)
  X_3x3 = BatchNormalization()(X_3x3)

  X_5x5 = Conv2D(num_filters[3], (1, 1), activation='relu')(X)
  X_5x5 = Conv2D(num_filters[4], (5, 5),
                 activation='relu', padding='same')(X_5x5)
  X_5x5 = BatchNormalization()(X_5x5)

  X_pool = MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(
      X) if maxpool else l2_norm2d(X, pool_size=(3, 3), strides=(2, 2), padding='same')
  X_pool = Conv2D(num_filters[5], (1, 1), activation='relu')(X_pool)
  X_pool = BatchNormalization()(X_pool)
  X_pool = ZeroPadding2D(7)(X_pool)

  X = Concatenate()([X_1x1, X_3x3, X_5x5, X_pool])

  return X

def block_3c(X):
  X_3x3 = Conv2D(128, (1, 1), activation='relu')(X)
  X_3x3 = Conv2D(256, (3, 3), strides=2,
                 activation='relu', padding='same')(X_3x3)
  X_3x3 = BatchNormalization()(X_3x3)

  X_5x5 = Conv2D(32, (1, 1), activation='relu')(X)
  X_5x5 = Conv2D(64, (5, 5), strides=2,
                 activation='relu', padding='same')(X_5x5)
  X_5x5 = BatchNormalization()(X_5x5)

  X_pool = MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(X)

  X = Concatenate()([X_3x3, X_5x5, X_pool])

  return X

def block_4abcd(X, num_filters):
  X_1x1 = Conv2D(num_filters[0], (1, 1), activation='relu')(X)
  X_1x1 = BatchNormalization()(X_1x1)

  X_3x3 = Conv2D(num_filters[1], (1, 1), activation='relu')(X)
  X_3x3 = Conv2D(num_filters[2], (3, 3),
                 activation='relu', padding='same')(X_3x3)
  X_3x3 = BatchNormalization()(X_3x3)

  X_5x5 = Conv2D(num_filters[3], (1, 1), activation='relu')(X)
  X_5x5 = Conv2D(num_filters[4], (5, 5),
                 activation='relu', padding='same')(X_5x5)
  X_5x5 = BatchNormalization()(X_5x5)

  X_pool = l2_norm2d(X, pool_size=(3, 3), strides=(2, 2), padding='same')
  X_pool = Conv2D(128, (1, 1), activation='relu')(X_pool)
  X_pool = BatchNormalization()(X_pool)
  X_pool = ZeroPadding2D(((3, 4), (3, 4)))(X_pool)

  X = Concatenate()([X_1x1, X_3x3, X_5x5, X_pool])

  return X

def block_4e(X):
  X_3x3 = Conv2D(160, (1, 1), activation='relu')(X)
  X_3x3 = Conv2D(256, (3, 3), strides=2,
                 activation='relu', padding='same')(X_3x3)
  X_3x3 = BatchNormalization()(X_3x3)

  X_5x5 = Conv2D(64, (1, 1), activation='relu')(X)
  X_5x5 = Conv2D(128, (5, 5), strides=2,
                 activation='relu', padding='same')(X_5x5)
  X_5x5 = BatchNormalization()(X_5x5)

  X_pool = MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(X)

  X = Concatenate()([X_3x3, X_5x5, X_pool])

  return X

def block_5ab(X, maxpool=True):
  X_1x1 = Conv2D(384, (1, 1), activation='relu')(X)
  X_1x1 = BatchNormalization()(X_1x1)

  X_3x3 = Conv2D(192, (1, 1), activation='relu')(X)
  X_3x3 = Conv2D(384, (3, 3), activation='relu', padding='same')(X_3x3)
  X_3x3 = BatchNormalization()(X_3x3)

  X_5x5 = Conv2D(48, (1, 1), activation='relu')(X)
  X_5x5 = Conv2D(128, (5, 5), activation='relu', padding='same')(X_5x5)
  X_5x5 = BatchNormalization()(X_5x5)

  X_pool = MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(
      X) if maxpool else l2_norm2d(X, pool_size=(3, 3), strides=(2, 2), padding='same')
  X_pool = Conv2D(128, (1, 1), activation='relu')(X_pool)
  X_pool = BatchNormalization()(X_pool)
  X_pool = ZeroPadding2D(((1, 2), (1, 2)))(X_pool)

  X = Concatenate()([X_1x1, X_3x3, X_5x5, X_pool])

  return X

def inception_network():
  X_input = Input((224, 224, 3))

  X = Conv2D(64, (7, 7), strides=(2, 2), activation='relu',
             padding='same', data_format='channels_last')(X_input)
  X = BatchNormalization()(X)
  X = MaxPool2D(pool_size=(3, 3), strides=2, padding='same')(X)

  X = block_2(X)
  X = block_3ab(X, [64, 96, 128, 16, 32, 32])
  X = block_3ab(X, [64, 96, 128, 32, 64, 64], False)
  X = block_3c(X)
  X = block_4abcd(X, [256, 96, 192, 32, 64])
  X = block_4abcd(X, [224, 112, 224, 32, 64])
  X = block_4abcd(X, [192, 128, 256, 32, 64])
  X = block_4abcd(X, [160, 144, 288, 32, 64])
  X = block_4e(X)
  X = block_5ab(X, False)
  X = block_5ab(X)

  X = AveragePooling2D(pool_size=(7, 7), strides=1)(X)
  X = Flatten()(X)
  X = Dense(128, name='dense_layer')(X)
  X = Lambda(lambda x: K.l2_normalize(x, axis=1))(X)

  return Model(inputs=X_input, outputs=X)
