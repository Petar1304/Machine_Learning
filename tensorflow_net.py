import numpy as np
import tensorflow as tf
import keras

# load data
mnist = keras.datasets.mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = tf.keras.utils.normalize(X_train, axis=1)
X_test = tf.keras.utils.normalize(X_test, axis=1)

# build model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=5)

# evaluate model
val_loss, val_acc = model.evaluate(X_test[:len(y_test)], y_test)

print(val_loss, val_acc)

# save and load model
model.save('mnist_net.model')
new_model = tf.keras.models.load_model('mnist_net.model')

# use model
predictions = model.predict([x_test])
print(np.argmax(predictions[5]))

