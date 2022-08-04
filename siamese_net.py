import tensorflow as tf
import tensorflow.keras.backend as k
import numpy as np
import cv2
import os

from config import EPOCHS, LEARNING_RATE, IMG_SIZE, VALIDATION_SPLIT

'''
Implementation of the siamese network to compare images of flies
'''

def load_data(path, label):
  '''
  Loading all images from folder 'path' and ading labels
  '''
  data = []
	# looping through all files in directory	
  for file in os.listdir(path):
    image_with_label = []
    img = cv2.imread(os.path.join(path, file))
    img = preprocessing(img)
    image_with_label.append(img)
    image_with_label.append(np.array(label)) # adding labels
    data.append(image_with_label)
  return data


def load_imgs(path):
  '''
  Loading images only
  '''
  data = []
	# looping through all files in directory	
  for file in os.listdir(path):
    img = cv2.imread(os.path.join(path, file))
    img = preprocessing(img)
    data.append(img)
  return data


def preprocessing(img):
  '''
  Resizing images and scaling them between 0 and 1
  '''
  img = cv2.resize(img, [IMG_SIZE, IMG_SIZE])
  img = img.astype(np.float32) / 255.0
  return img


def make_training_data(data):
  '''
  Shuffling data and seperating images from labels
  '''
  np.random.shuffle(data)
  X = np.array([item[0] for item in data], dtype=np.float32)
  y = np.array([item[1] for item in data], dtype=np.uint8)
  return X, y


# IMPLEMENTATION OF SIAMESE NEURAL NETWORK
def create_siamese_model():
  '''
  outputs 128 dimensional vector
  '''
  
  # Inputs
  inputs = tf.keras.layers.Input((IMG_SIZE, IMG_SIZE, 3))
 
  # Conv layers
  x = tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), padding='same', activation='relu')(inputs)
  x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
  x = tf.keras.layers.Dropout(0.3)(x)

  x = tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu')(x)
  x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
  x = tf.keras.layers.Dropout(0.3)(x)

  x = tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu')(x)
  x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
  x = tf.keras.layers.Dropout(0.3)(x)

  # Fully connected layers
  pooledOutput = tf.keras.layers.GlobalAveragePooling2D()(x)
  pooledOutput = tf.keras.layers.Dense(1024)(pooledOutput)
  outputs = tf.keras.layers.Dense(128)(pooledOutput)

  model = tf.keras.Model(inputs, outputs)
  return model


def euclidean_distance(vectors):
  '''
  Calculating euclidean distance between 2 features vectors which are outputs of network 
  '''
  (featA, featB) = vectors
  sum_squared = k.sum(k.square(featA - featB), axis=1, keepdims=True)
  return k.sqrt(k.maximum(sum_squared, k.epsilon()))


def generate_train_image_pairs(data, pos_imgs, neg_imgs):
  '''
  Generating positive (2 images from same class) and negative (2 images from different classes)
  image pairs for training siamese network
  '''
  pair_images = []
  pair_labels = []
  
  for img, label in data:

    if (label == 1):
      # positive pair
      pos_img = pos_imgs[np.random.choice(len(pos_imgs))] # choose random index
      pair_images.append((img, pos_img))
      pair_labels.append(1)	
    
      # negative pair
      neg_img = neg_imgs[np.random.choice(len(neg_imgs))]
      pair_images.append((img, neg_img))
      pair_labels.append(0)

    if (label == 0):
      # positive pair
      pos_img = neg_imgs[np.random.choice(len(neg_imgs))]
      pair_images.append((img, pos_img))
      pair_labels.append(1)

      # negative pair
      neg_img = pos_imgs[np.random.choice(len(pos_imgs))]
      pair_images.append((img, neg_img))
      pair_labels.append(0)

  return np.array(pair_images), np.array(pair_labels)


if __name__ == '__main__':
  
  # loading data
  positive_imgs_path = '/home/petar/Zentrixlab/data/training_data/'
  negative_imgs_path = '/home/petar/Zentrixlab/data/training_data_other/'

  data_positive = load_data(positive_imgs_path, 1)
  data_negative = load_data(negative_imgs_path, 0)

  data = data_positive + data_negative
  np.random.shuffle(data)

  imgs_pos = load_imgs(positive_imgs_path)
  imgs_neg = load_imgs(negative_imgs_path)

  images_pair, labels_pair = generate_train_image_pairs(data, imgs_pos, imgs_neg)

  # extracting features with cnn
  feature_extractor = create_siamese_model()

  imgA = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
  imgB = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
 
  featA = feature_extractor(imgA)
  featB = feature_extractor(imgB)
 
  # calculating distance between output vectors
  distance = tf.keras.layers.Lambda(euclidean_distance)((featA, featB))
  
  outputs = tf.keras.layers.Dense(1, activation="sigmoid")(distance)
  
  model = tf.keras.Model(inputs=[imgA, imgB], outputs=outputs)
  
  model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), metrics=["accuracy"])

  # training
  history = model.fit([images_pair[:][0], images_pair[:][1]], labels_pair[:], validation_split=VALIDATION_SPLIT, batch_size=4, epochs=EPOCHS)


