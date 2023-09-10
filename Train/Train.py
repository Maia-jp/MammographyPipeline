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

from ..Util import safe_make_folder
from .Generator import CustomGenerator, teste

teste()

def train_UNET(dataset_folder = os.environ["DATASET_FOLDER"]):

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

            if(len(gpus) > 1 ):
                #mirrored_strategy = tf.distribute.MirroredStrategy()
                tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


    sm.set_framework('tf.keras')
    sm.framework()

    execution_name = "execution_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%p")
    training_results_folder =  os.environ["RESULTS_FOLDER"] + execution_name
    safe_make_folder(training_results_folder)

    image_folder=os.path.join(dataset_folder,"image")
    label_folder=os.path.join(dataset_folder,"label")
    training_csv_filename=os.path.join(dataset_folder,"training.csv")
    validation_csv_filename=os.path.join(dataset_folder,"validation.csv")

    train_generator = CustomGenerator(training_csv_filename, image_folder, label_folder, shuffle=True, use_augmentation=True)
    validation_generator = CustomGenerator(validation_csv_filename, image_folder, label_folder, shuffle=False, use_augmentation=False)

    model = sm.Unet('efficientnetb3', input_shape=(None, None, 1), encoder_weights=None, classes=5, activation='softmax')

    model.compile('Adam',loss=sm.losses.categorical_focal_jaccard_loss,metrics=[sm.metrics.iou_score])


    early_stop = EarlyStopping(monitor='val_loss', patience=30)
    model_checkpoint = ModelCheckpoint(os.path.join(training_results_folder,'weights.h5'), save_weights_only=True, save_best_only=True, mode='auto')
    csv_logger = CSVLogger(os.path.join(training_results_folder,'history.csv'))
    callbacks_list = [early_stop, model_checkpoint, csv_logger]

    model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=200,
            validation_data=validation_generator,
            validation_steps=len(validation_generator),
            callbacks=callbacks_list,
            shuffle=True,
            verbose=1)

    model.load_weights(os.path.join(training_results_folder,'weights.h5'))

    onnx_model_ss_structures_of_interest = onnxmltools.convert_keras(model, target_opset=12)
    onnxmltools.utils.save_model(onnx_model_ss_structures_of_interest, os.path.join(training_results_folder,'model.onnx'))