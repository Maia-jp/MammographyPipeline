import copy
import os
import random
import re
from datetime import datetime

import cv2
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage.io as io
from skimage import draw, exposure, img_as_float, img_as_uint
import tensorflow as tf
from tensorflow.keras.callbacks import History, EarlyStopping, ModelCheckpoint, CSVLogger

import onnxmltools
import segmentation_models as sm

import tf2onnx


from .Generator import CustomGenerator
from ..Util.Util import safe_make_folder
from ..Util import SQLogger

def train_UNET(dataset_folder = os.environ["DATASET_FOLDER"]):
    logger = SQLogger.ExperimentLogger(os.environ["DBLOG_FOLDER"])

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

    logger.create_experiment(execution_name, 'UNET_efficientnetb3', os.path.basename(os.path.normpath(dataset_folder)))

    model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=int(os.environ["EPOCHS"]),
            validation_data=validation_generator,
            validation_steps=len(validation_generator),
            callbacks=callbacks_list,
            shuffle=True,
            verbose=1)
    
    # Append the contents of history.csv to the History table
    experiment_id = execution_name
    with open(os.path.join(training_results_folder, 'history.csv'), 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            epoch = int(row['epoch'])
            iou_score = float(row['iou_score'])
            loss = float(row['loss'])
            val_iou_score = float(row['val_iou_score'])
            val_loss = float(row['val_loss'])
            # Log the history entry into the History table
            logger.log_history(experiment_id, epoch, iou_score, loss, val_iou_score, val_loss)

    logger.close()
    model.load_weights(os.path.join(training_results_folder,'weights.h5'))

    onnx_model_ss_structures_of_interest = onnxmltools.convert_keras(model, target_opset=12)
    onnxmltools.utils.save_model(onnx_model_ss_structures_of_interest, os.path.join(training_results_folder,'model.onnx'))
    return model, os.path.join(training_results_folder,'model.onnx')




def Train_StylizedDataset(model, dataset_folder: str):
    logger = SQLogger.ExperimentLogger(os.environ["DBLOG_FOLDER"])
    
    import onnxruntime as ort
    # model = ort.InferenceSession('data/results/training/execution_2023_08_07_19_06_39_PM/model.onnx', providers=['CUDAExecutionProvider']) #loading model
    # model = ort.InferenceSession(modelPath, providers=['CUDAExecutionProvider']) #loading model
    # model_input_name = model.get_inputs()[0].name #getting input name for the model
    # model.run(None, {model_input_name: np.zeros((1,384,384,1),dtype=np.float32)})
    
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

    execution_name =  "retrained_execution_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%p")
    training_results_folder =  os.environ["RESULTS_FOLDER"] + execution_name
    safe_make_folder(training_results_folder)

    image_folder=os.path.join(dataset_folder,"image")
    label_folder=os.path.join(dataset_folder,"label")
    training_csv_filename=os.path.join(dataset_folder,"training.csv")
    validation_csv_filename=os.path.join(dataset_folder,"validation.csv")

    train_generator = CustomGenerator(training_csv_filename, image_folder, label_folder, shuffle=True, use_augmentation=True)
    validation_generator = CustomGenerator(validation_csv_filename, image_folder, label_folder, shuffle=False, use_augmentation=False)

    # Compile the model with the previous configuration
    model.compile('Adam', loss=sm.losses.categorical_focal_jaccard_loss, metrics=[sm.metrics.iou_score])

    # Define callbacks for the new training
    new_callbacks_list = [EarlyStopping(monitor='val_loss', patience=30),
                          ModelCheckpoint(os.path.join(training_results_folder, 'weights.h5'), save_weights_only=True, save_best_only=True, mode='auto'),
                          CSVLogger(os.path.join(training_results_folder, 'history.csv'))]

    # Log the new experiment
    logger.create_experiment(execution_name, 'UNET_efficientnetb3', os.path.basename(os.path.normpath(dataset_folder)))

    # Fit the model with the new dataset
    model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=int(os.environ["EPOCHS"]),
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=new_callbacks_list,
        shuffle=True,
        verbose=1
    )

    # Similar to what you did in train_UNET, log the history
    experiment_id = execution_name
    with open(os.path.join(training_results_folder, 'history.csv'), 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            epoch = int(row['epoch'])
            iou_score = float(row['iou_score'])
            loss = float(row['loss'])
            val_iou_score = float(row['val_iou_score'])
            val_loss = float(row['val_loss'])
            # Log the history entry into the History table
            logger.log_history(experiment_id, epoch, iou_score, loss, val_iou_score, val_loss)

    # Save the retrained model to ONNX format
    model.load_weights(os.path.join(training_results_folder, 'weights.h5'))
    onnx_model_ss_structures_of_interest = onnxmltools.convert_keras(model, target_opset=12)
    onnxmltools.utils.save_model(onnx_model_ss_structures_of_interest, os.path.join(training_results_folder, 'model.onnx'))

    # Return the retrained model and its ONNX representation
    return model, os.path.join(training_results_folder, 'model.onnx')
