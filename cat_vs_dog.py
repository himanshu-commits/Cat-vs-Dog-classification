
from google.colab import drive
drive.mount('/content/drive/')

import os

train_dog_dir = os.path.join('/content/drive/MyDrive/cats_and_dogs_filtered/train/dogs')
train_cat_dir = os.path.join('/content/drive/MyDrive/cats_and_dogs_filtered/train/cats')
validation_dog_dir = os.path.join('/content/drive/MyDrive/cats_and_dogs_filtered/validation/dogs')
validation_cat_dir = os.path.join('/content/drive/MyDrive/cats_and_dogs_filtered/validation/cats')

train_dog = os.listdir(train_dog_dir)
train_cat = os.listdir(train_cat_dir)
validation_dog = os.listdir(validation_dog_dir)
validation_cat = os.listdir(validation_cat_dir)
print(train_dog[:10])
print(train_cat[:10])
print(validation_dog[:10])
print(validation_cat[:10])

print('total training cat images:', len(os.listdir(train_cat_dir)))
print('total training dog images:', len(os.listdir(train_dog_dir)))
print('total validation cat images:', len(os.listdir(validation_cat_dir)))
print('total validation dog images:', len(os.listdir(validation_dog_dir)))

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

nrows = 4
ncols = 4

pic_index = 0

fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_cat_pix = [os.path.join(train_cat_dir, fname) 
                for fname in train_cat[pic_index-8:pic_index]]
next_dog_pix = [os.path.join(train_dog_dir, fname) 
                for fname in train_dog[pic_index-8:pic_index]]

for i, img_path in enumerate(next_cat_pix+next_dog_pix):
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') 

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

import tensorflow as tf

model = tf.keras.models.Sequential(
    [
     tf.keras.layers.Conv2D(16,(3,3),input_shape=(150,150,3),activation = 'relu'),
     tf.keras.layers.MaxPooling2D(2,2),
     tf.keras.layers.Conv2D(34,(3,3),activation='relu'),
     tf.keras.layers.MaxPooling2D(2,2),
     tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
     tf.keras.layers.MaxPooling2D(2,2),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(512,activation='relu'),
     tf.keras.layers.Dense(1,activation='sigmoid')
    ]
)

model.summary()

from tensorflow.keras.optimizers import RMSprop
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate = 0.001),
              metrics = ['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
    )
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    '/content/drive/MyDrive/cats_and_dogs_filtered/train',
    target_size = (150,150),
    batch_size = 20,
    class_mode = 'binary'
)
validation_generator = test_datagen.flow_from_directory(
    '/content/drive/MyDrive/cats_and_dogs_filtered/validation',
    target_size = (150,150),
    batch_size = 20,
    class_mode = 'binary'
)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=15,
    validation_data = validation_generator,
    validation_steps = 50,
    verbose=2
)

#Retrieve a list of accuracy results on training and validation data
# sets for each training epoch
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Retrieve a list of list results on training and validation data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')

plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')

import numpy as np
from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()
for fn in uploaded.keys():
  path = '/content/'+fn
  img = image.load_img(path,target_size = (150,150))
  x = image.img_to_array(img)
  x = np.expand_dims(x,axis = 0)

  images = np.vstack([x])
  classes = model.predict(images,batch_size=10)
  print(classes[0])
  if classes[0]>0:
    print(fn + " is a dog")
  else:
    print(fn + " is a cat")

