# Classify an image to Dog's or Cat's using Convolutional Neural Network

#------------------------------------------------------------------
#Importing the libraries

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#------------------------------------------------------------------
#Step 1 - Data Preprocessing

image_w = 64
image_h = 64
image_d = 3

##Preprocessing the Training set
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (image_w, image_h),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


##Preprocessing the Test set
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (image_w, image_h),
                                            batch_size = 32,
                                            class_mode = 'binary')

#------------------------------------------------------------------
# Step 2 - Building the CNN

##Initialising the CNN
cnn = tf.keras.models.Sequential()

# Layer 1 - convolution layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[image_w, image_h, image_d]))

# Layer 2 - max pooling layer
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Layer 3 - second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu'))

# Layer 4 - second max pooling layer 
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Flattening
cnn.add(tf.keras.layers.Flatten())

# Layer 5 - Full Connected layer
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Layer 6 - Output layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

#------------------------------------------------------------------
# Step 3 - Training and Testing the CNN model

## Compiling the model
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

## Training the model on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 50)

