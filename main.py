from PIL import Image
from PIL import ImageFilter
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
import keras
import tensorflow as tf
import os
import time

######## MAIN #######


img_rows= 32
img_cols= 32

def load_data_from_category(category_path, label):
    data = []
    labels = []
    for image_name in os.listdir(category_path):
        image_path = os.path.join(category_path, image_name)
        if not image_name.endswith('.jpeg'):
            continue
        img = Image.open(image_path)
        img = img.convert('L')
        img = img.resize((img_rows, img_cols))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=-1)
        data.append(img_array)
        labels.append(label)
    data= np.array(data)
    labels= np.array(labels)

    return data, labels


folder_path = "chest_xray"
normal_category_path_train = os.path.join(folder_path, "train", "NORMAL")
data_normal_train, labels_normal_train = load_data_from_category(normal_category_path_train, 0)
normal_category_path_test = os.path.join(folder_path, "test", "NORMAL")
data_normal_test, labels_normal_test = load_data_from_category(normal_category_path_test, 0)
normal_category_path_val = os.path.join(folder_path, "val", "NORMAL")
data_normal_val, labels_normal_val = load_data_from_category(normal_category_path_val, 0)

pneumonia_category_path_train = os.path.join(folder_path, "train", "PNEUMONIA")
data_pneumonia_train, labels_pneumonia_train = load_data_from_category(pneumonia_category_path_train, 1)
data_pneumonia_path_test = os.path.join(folder_path, "test", "PNEUMONIA")
data_pneumonia_test, labels_pneumonia_test = load_data_from_category(data_pneumonia_path_test, 1)
data_pneumonia_path_val = os.path.join(folder_path, "val", "PNEUMONIA")
data_pneumonia_val, labels_pneumonia_val = load_data_from_category(data_pneumonia_path_val, 1)


data_train = np.concatenate((data_normal_train, data_pneumonia_train))
labels_train = np.concatenate((labels_normal_train, labels_pneumonia_train))

data_test = np.concatenate((data_normal_test, data_pneumonia_test))
labels_test = np.concatenate((labels_normal_test, labels_pneumonia_test))

data_val = np.concatenate((data_normal_val, data_pneumonia_val))
labels_val = np.concatenate((labels_normal_val, labels_pneumonia_val))
num_classes = 2



model = tf.keras.models.Sequential()

#model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='valid', input_shape=[1,32,32,3]))#Convolution
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), padding='valid'))#Convolution
model.add(tf.keras.layers.Activation('relu'))#Activation function
model.add(tf.keras.layers.AveragePooling2D(pool_size=(30, 30)))#30x30 average pooling
model.add(tf.keras.layers.Flatten())# shape equals to [batch_size, 32] 32 is the number of filters
model.add(tf.keras.layers.Dense(1))#Fully connected layer
model.add(tf.keras.layers.Activation('sigmoid'))
opt= keras.optimizers.Adam(learning_rate= 0.001)

model.compile(loss='binary_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

batch_size=10
epochs= 20
def train():
	model.fit(data_train, labels_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(data_val, labels_val),
              shuffle=True)

train()
start_time = time.time()
train()
end_time = time.time()

print(f"Training time: {end_time - start_time} seconds")
