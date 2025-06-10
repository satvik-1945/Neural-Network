import tensorflow as tf
from nn.models.Sequential import Sequential
from nn.layers.Dense import Dense
from nn.layers.Flatten import Flatten
 
mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()
 
x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)
 
print(f"Training on {len(x_train)} samples")
print(f"Input shape: {x_train.shape}")
 
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))
 
model.compile(optimizer='adam', metrics=['accuracy'])
 
model.fit(x_train, y_train, epochs=100, learning_rate=0.01, batch_size=32)
 
model.save('handwritten.pkl')