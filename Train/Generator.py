import copy
import os
import random
import re
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.io as io
from skimage import draw, exposure, img_as_float, img_as_uint
import tensorflow as tf
from tensorflow.keras.callbacks import History, EarlyStopping, ModelCheckpoint, CSVLogger

import onnxmltools
import segmentation_models as sm

from ..Util.Util import safe_make_folder

__all__ = ['CustomGenerator',
           'test_generator'
           ]

class CustomGenerator(tf.keras.utils.Sequence):
    def __init__(self, csv_filename, image_folder, label_folder, input_size=(384,384), batch_size=4, n_classes=5, shuffle=False, use_augmentation=False):
        """
        Constructor for the CustomGenerator class.

        Parameters:
        - csv_filename (str): Path to a CSV file containing image filenames and labels.
        - image_folder (str): Path to the folder containing the images.
        - label_folder (str): Path to the folder containing the label maps.
        - input_size (tuple): The size to which images and labels will be resized.
        - batch_size (int): Number of samples in each batch.
        - n_classes (int): Number of classes for one-hot encoding of labels.
        - shuffle (bool): Whether to shuffle the data during training.
        - use_augmentation (bool): Whether to apply data augmentation during training.
        """
        self.input_size = input_size
        self.batch_size = batch_size
        self.n_classes = n_classes
        raw_df = pd.read_csv(csv_filename)
        self.filenames = list(raw_df['filename'])
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.shuffle = shuffle
        self.use_augmentation = use_augmentation
        self.n_samples = len(self.filenames)
        self.indexes = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(self.indexes)

        # Image data augmentation settings using TensorFlow's ImageDataGenerator
        self.image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=0.0, width_shift_range=0.0, height_shift_range=0.0, zoom_range=0.1,
            horizontal_flip=False, brightness_range=(0.8, 1.2), shear_range=0.0,
            channel_shift_range=0, fill_mode='constant', cval=0
        )

        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.n_samples)
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.n_samples / self.batch_size))

    def normalize_image(self, img):
        'Image normalization'
        return img.astype(np.float32) / 255.0

    def augment_data(self, img, label_map):
        """
        Apply data augmentation to an image and its corresponding label map.

        Parameters:
        - img (numpy.ndarray): The image to augment.
        - label_map (numpy.ndarray): The corresponding label map.

        Returns:
        - Augmented image and label map.
        """
        transform_parameters = self.image_datagen.get_random_transform(img_shape=img.shape)

        t_img = img
        t_label_map = label_map.reshape((label_map.shape[0], label_map.shape[1], 1))
        t_img = self.image_datagen.apply_transform(t_img, transform_parameters)
        transform_parameters['brightness'] = None  # Ensure only Affine transformations are kept
        transform_parameters['channel_shift_intensity'] = None
        t_label_map = self.image_datagen.apply_transform(t_label_map, transform_parameters)
        img = t_img
        label_map = t_label_map[:, :, 0]
        return img, label_map

    def resize_data(self, img, label_map):
        'Resizes the input image and the input label image to the target size'
        size = (self.input_size[1], self.input_size[0])
        resized_img = cv2.resize(img, size, interpolation=cv2.INTER_NEAREST)
        resized_img = resized_img.reshape((resized_img.shape[0], resized_img.shape[1], 1))
        resized_label_map = cv2.resize(label_map, size, interpolation=cv2.INTER_NEAREST)
        return resized_img, resized_label_map

    def get_raw_batch_data(self, current_indexes):
        """
        Retrieve raw batch data without applying augmentation.

        Parameters:
        - current_indexes (list): Indexes of the batch.

        Returns:
        - List of images and label maps.
        """
        images = []
        label_maps = []

        for i in range(len(current_indexes)):
            img = io.imread(os.path.join(self.image_folder, self.filenames[current_indexes[i]]))
            img = img.reshape((img.shape[0], img.shape[1], 1))
            images.append(img)
            label_map = io.imread(os.path.join(self.label_folder, self.filenames[current_indexes[i]]))
            label_maps.append(label_map)

        return images, label_maps

    def __data_generation(self, current_indexes):
        'Generates data containing batch_size samples and applies the corresponding augmentations.'

        # Initialization
        current_batch_size = len(current_indexes)

        raw_images, raw_label_maps = self.get_raw_batch_data(current_indexes)

        X = np.zeros((current_batch_size, self.input_size[0], self.input_size[1], 1),
                     dtype=np.float32)
        y = np.zeros((current_batch_size, self.input_size[0], self.input_size[1], self.n_classes),
                     dtype=np.float32)

        # Generate data
        for i, idx in enumerate(current_indexes):
            # Store sample
            img = raw_images[i]
            label_map = raw_label_maps[i]
            if self.use_augmentation:
                img, label_map = self.augment_data(img, label_map)
            img, label_map = self.resize_data(img, label_map)
            img = self.normalize_image(img)
            X[i, :, :, :] = img

            y[i, :, :, :] = tf.keras.utils.to_categorical(label_map, self.n_classes)
        return X, y

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        current_indexes = [self.indexes[(index * self.batch_size + i) % len(self.indexes)] for i in range(self.batch_size)]
        # Generate data
        X, y = self.__data_generation(current_indexes)

        return X, y

    def __call__(self):
        """
        Generator function to yield batches of data. It iterates through all samples.
        """
        for i in self.indexes:
            yield self.__getitem__(i)


def test_generator(dataset_folder = os.environ["DATASET_FOLDER"]):

    image_folder=os.path.join(dataset_folder,"image")
    label_folder=os.path.join(dataset_folder,"label")
    # training_csv_filename=os.path.join(dataset_folder,"training.csv")
    csv_filename=os.path.join(dataset_folder,"validation.csv")
    # training_csv_filename=os.path.join(dataset_folder,"all_files.csv")



    gen_folder = os.environ["GENERATOR_FOLDER"]
    safe_make_folder(gen_folder)
    train_generator = CustomGenerator(csv_filename, image_folder, label_folder, shuffle=True, use_augmentation=False)
    for i in range(len(train_generator)):
        batchX, batchy = train_generator.__getitem__(i)
        fig, axs = plt.subplots(batchX.shape[0], 6)
        for j in range(batchX.shape[0]):
            axs[j,0].imshow(batchX[j,:,:,0], cmap='gray')
            axs[j,1].imshow(batchy[j,:,:,0], cmap='gray')
            axs[j,2].imshow(batchy[j,:,:,1], cmap='gray')
            axs[j,3].imshow(batchy[j,:,:,2], cmap='gray')
            axs[j,4].imshow(batchy[j,:,:,3], cmap='gray')
            axs[j,5].imshow(batchy[j,:,:,4], cmap='gray')
        fig.savefig(os.path.join(gen_folder,str(i)))
        plt.close(fig)



def teste():
    print("here")