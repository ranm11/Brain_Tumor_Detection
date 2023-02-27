from loadAndPreprocess import LoadAndPreProcess
import matplotlib.pyplot as plt
import keras.utils as image
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout ,MaxPooling2D,Conv2D,Flatten,Dense

def build_Model(loadInstance):
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(loadInstance.IMAGE_SIZE, loadInstance.IMAGE_SIZE, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def build_model(loadInstance):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(loadInstance.IMAGE_SIZE, loadInstance.IMAGE_SIZE, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

loadInstance = LoadAndPreProcess('..\\archive\\yes', '..\\archive\\no','../')
#data to plot if need
[train_data,test_data,train_label,test_label] = loadInstance.CreateFolderDataset()

#data to analyze 
[train_data,test_data,train_label,test_label] = loadInstance.getNormalizeData()
train_data.shape
#build_Model(loadInstance)


model = build_model(loadInstance);
history = model.fit(train_data, train_label, epochs=85, validation_split=0.2, verbose=1)
result =  model.predict(test_data)
