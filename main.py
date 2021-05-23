import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# ################PARAMETERS########################
path = 'Data/trainingSet/trainingSet'
x = []
y = []
data_shape = (28, 28)
noOfFilters = 60
filterSize1 = (5, 5)
filterSize2 = (3, 3)
poolingSize = (2, 2)
noOfNode = 500
####################################################

# list of classes
classes = os.listdir(path)
print(classes)


# pre processing function for images
def pre_processing(img):
    processed = cv2.resize(img, data_shape)
    processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    processed = cv2.equalizeHist(processed)
    processed = processed/255
    processed = np.array(processed)
    return processed


# putting images and labels into a matrix
print('processing images...')
for i in range(10):
    img_lst = os.listdir(path+'/'+str(i))
    print(str(i), end=' ')
    for n in range(len(img_lst)):
        img = img_lst[n]
        new_img = cv2.imread(path+'/'+str(i)+'/'+img, 1)
        new_img = pre_processing(new_img)
        x.append(new_img)
        y.append(i)

# reshaping data to enter the cnn
x = np.array(x)
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)
y = np.array(y)

# splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# creates varaity in the pictures
Gen = ImageDataGenerator(width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2,
                         shear_range=0.1,
                         rotation_range=10)
Gen.fit(x_train)

# transform labels to One-Hot encoding
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# model structure
model = Sequential()
model.add((Conv2D(noOfFilters, filterSize1, input_shape=(28, 28, 1), activation='relu')))
model.add((Conv2D(noOfFilters, filterSize1, activation='relu')))
model.add((MaxPooling2D(pool_size=poolingSize)))
model.add((Conv2D(noOfFilters//2, filterSize2, activation='relu')))
model.add((Conv2D(noOfFilters//2, filterSize2, activation='relu')))
model.add((MaxPooling2D(pool_size=poolingSize)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(noOfNode, activation='relu'))
model.add((Dense(10, activation='softmax')))

# defying optimizer
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# start training
model.fit_generator(Gen.flow(x_train, y_train), epochs=5)

# gives score to the model
score = model.evaluate(x_test, y_test)
print('test score:', score[0])
print('test acc:', score[1])

# saving the model to a file
model.save('trained_model.h5')