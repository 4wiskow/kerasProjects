"""
Recognition of handwritten digits based on MNIST data set.
"""

from keras.datasets import mnist
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils import np_utils


def main(x_train, x_test, y_train, y_test):
    model = Sequential()
    model.add(Dense(units=100, input_dim=784))
    model.add(Activation('sigmoid'))
    model.add(Dense(units=10))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=1000, epochs=100)

    _eval = model.evaluate(x_test, y_test, batch_size=1000)
    print _eval

    return model


def preprocessing(x_train, y_train, x_test, y_test):
    num_pixels = x_train.shape[1] * x_train.shape[2]
    x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
    x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')

    # normalize
    x_train = x_train/255
    x_test = x_test/255

    # one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    return x_train, x_test, y_train, y_test

if __name__ == '__main__':
    (X_TRAIN, Y_TRAIN), (X_TEST, Y_TEST) = mnist.load_data()
    X_TRAIN, Y_TRAIN, X_TEST, Y_TEST = preprocessing(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST)
    MODEL = main(X_TRAIN, Y_TRAIN, X_TEST, Y_TEST)

    # plt.imshow(X_TRAIN[1000], cmap=plt.get_cmap('gray'))
    # plt.show()
    # _class = model.predict(np.expand_dims(x_train[1], axis=0), batch_size=1)
    # print np.argmax(_class)
