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



import tensorflow as tf
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation, Trainer, TrainingArguments
from torch.utils.data import Dataset
from .Generator import SegformerDataset
def train_segformer(dataset_folder=os.environ["DATASET_FOLDER"]):
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

    execution_name = "execution_" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%p")
    training_results_folder = os.environ["RESULTS_FOLDER"] + execution_name
    safe_make_folder(training_results_folder)

    image_folder=os.path.join(dataset_folder,"image")
    label_folder=os.path.join(dataset_folder,"label")
    training_csv_filename=os.path.join(dataset_folder,"training.csv")
    validation_csv_filename=os.path.join(dataset_folder,"validation.csv")

    # Define feature extractor, model, and augmentation
    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    feature_extractor = SegformerFeatureExtractor.from_pretrained(model_name)
    model = SegformerForSemanticSegmentation.from_pretrained(model_name, num_labels=5)
    # augmentation = ...  # Define your augmentation function here TODO

    # Create datasets
    train_dataset = SegformerDataset(training_csv_filename, image_folder, label_folder, feature_extractor)
    validation_dataset = SegformerDataset(validation_csv_filename, image_folder, label_folder, feature_extractor)

    training_args = TrainingArguments(
        output_dir=training_results_folder,  # Output directory
        do_train=True,  # Run training
        do_eval=True,  # Run evaluation on the validation set
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        per_device_train_batch_size=8,  # Batch size per device during training
        per_device_eval_batch_size=8,  # Batch size per device during evaluation
        num_train_epochs=int(os.environ["EPOCHS"]),  # Total number of training epochs
        # (Add other essential arguments as needed, such as learning rate, optimizer, etc.)
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        # compute_metrics=compute_metrics,
    )

    # Start training
    trainer.train()

    # ... (Log training results as needed) ...

    # Save the model
    model.save_pretrained(training_results_folder)  # Saves in Hugging Face format
    # (Optional) Convert and save the model in ONNX format
    # ...

    return model, training_results_folder  # Return the model and path to saved model
