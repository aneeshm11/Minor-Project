import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Aneesh\Downloads\CODE\project\Crystal_structure.csv")
df = df.drop(['In literature', 'Compound'], axis=1)
df_cleaned = df.fillna(0)
df_cleaned = df_cleaned.replace('-', 0)
df_encoded = pd.get_dummies(df_cleaned, columns=['A', 'B', 'Lowest distortion'])
dfn = df_encoded.drop("Lowest distortion_0", axis=1)
xtrain = dfn.iloc[:, :159]
ytrain = dfn.iloc[:, 159:]
xtrain = np.array(xtrain, dtype=np.float32)
ytrain = np.array(ytrain, dtype=np.float32)

model = Sequential()
model.add(Dense(300, input_dim=159, activation='relu', kernel_initializer=glorot_uniform()))
model.add(Dense(200, activation='relu', kernel_initializer=glorot_uniform()))
model.add(Dense(120, activation='sigmoid', kernel_initializer=glorot_uniform()))
model.add(Dense(60, activation='relu', kernel_initializer=glorot_uniform()))
model.add(Dense(4, activation='softmax', kernel_initializer=glorot_uniform()))

initial_lr = 0.01

def lr_schedule(epoch):
    lr = initial_lr * np.exp(-0.1 * epoch)
    return lr

adam_optimizer = Adam(learning_rate=initial_lr)
model.compile(optimizer=adam_optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

lr_scheduler = LearningRateScheduler(lr_schedule)

history = model.fit(xtrain, ytrain, epochs=120, batch_size=50, callbacks=[lr_scheduler])

train_loss, train_acc = model.evaluate(xtrain, ytrain)
print(f'Training Accuracy: {train_acc * 100:.2f}%')


model.save("data.h5")

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epochs')
plt.legend()
plt.tight_layout()
plt.savefig('results.jpeg')
plt.show()