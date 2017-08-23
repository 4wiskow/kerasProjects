"""
Multi-Layer Perceptron capable of learning simple boolean operators.
"""
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras import optimizers

X_TRAIN = [[0, 0], [0, 1], [1, 0], [1, 1]]
Y_TRAIN = [[1], [0], [0], [1]]

MODEL = Sequential()

#  Stack layers
MODEL.add(Dense(units=3, input_dim=2))
MODEL.add(Activation('softsign'))
MODEL.add(Dense(units=1, input_dim=3))

OPTIMIZER = optimizers.SGD(lr=0.2)

#  Configure Model
MODEL.compile(loss='mean_absolute_error',
              optimizer=OPTIMIZER,
              metrics=['accuracy'])

#  Fit model
MODEL.fit(X_TRAIN, Y_TRAIN, epochs=150, batch_size=4)

#  Evaluate
LOSS_AND_METRICS = MODEL.evaluate(X_TRAIN, Y_TRAIN, batch_size=4)
print LOSS_AND_METRICS

#  Predict
CLASSES = MODEL.predict(X_TRAIN, batch_size=4)
print CLASSES
