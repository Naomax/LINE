import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from keras.preprocessing.text import Tokenizer

def my_func(arg):
  arg = tf.convert_to_tensor(arg, dtype=tf.float32)
  return arg

dic_line=0
dim_in=50
line=0
fi=open("dictionary.txt","r",encoding="utf-8")
for row in fi:
    dic_line=dic_line+1
fi.close()

inputs = keras.Input(shape=(dim_in,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = layers.Dense(dic_line, activation="softmax", name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=o1,o2,o3,o4,o5,o6,o7,o8,o9,o10,o11,o12,o13,o14,o15)

#(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#print(x_train)#28*28の二次元配列が70000個
#x_train=#25個の要素からなる１次元配列が25000個。
with open("dictionary.txt","r",encoding="utf-8") as fi:
    content=fi.readlines()
    dictionary_list=[x.strip() for x in content]

fi=open("input2.txt","r",encoding="utf-8")
for row in fi:
    line=line+1
x=[[0]*dim_in for i in range(line)]
fi.close()

fi=open("input2.txt","r",encoding="utf-8")
fo=open("output2.txt","r",encoding="utf-8")
content=fi.readlines()
content2=fo.readlines()
tokenizer=Tokenizer()
tokenizer.fit_on_texts(content)
tokenizer.fit_on_texts(content2)
sequence=tokenizer.texts_to_sequences(content)
sequence2=tokenizer.texts_to_sequences(content2)
#print(content)
length=len(dictionary_list)
fi.close()
fo.close()

#x_train = x_train.reshape(60000, 784).astype("float32") / 255 #第二引数の要素からなる一次元配列を第一引数個生成
x_train=[[0]*dim_in for i in range(33361)]
for i in range(33361):
    for j in range(min(dim_in,len(sequence[i]))):
        x_train[i][j]=sequence[i][j]        
#x_test = x_test.reshape(10000, 784).astype("float32") / 255 #784個の要素からなる10000個のテストデータ
#x_train = my_func(x_train)
x_train=np.array(x_train)
x_train=my_func(x_train)
x_val=x_train[-3000:]
x_train=x_train[:-3000]

y_train=[[0]*dim_in for i in range(33361)]
for i in range(33361):
    for j in range(min(dim_in,len(sequence2[i]))):
        y_train[i][j]=sequence2[i][j]
#y_train=my_func(y_train)
y_train=np.array(y_train)
y_train=my_func(y_train)
y_val=y_train[-3000:]
y_train=y_train[:-3000]
#y_train = y_train.astype("float32")
#y_test = y_test.astype("float32")

model.compile(
    optimizer=keras.optimizers.RMSprop(),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),
    # List of metrics to monitor
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

#print("Fit model on training data")
history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=10,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_val, y_val),
)

#history.history
#print(x_train,y_train)
# Preprocess the data (these are NumPy arrays)