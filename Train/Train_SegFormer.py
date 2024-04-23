import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from datasets import load_metric
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
import random
import pandas as pd


from .Generator_SegFormerDataset import SemanticSegmentationDataset
from .Generator_SegFormerFinetuner import SegformerFinetuner
from ..Util.Util import safe_make_folder
from ..Util import SQLogger


def train_UNET(dataset_folder = os.environ["DATASET_FOLDER"]):
    feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
    feature_extractor.do_reduce_labels = False
    feature_extractor.size = 128
    dataset = "/content/drive/MyDrive/Faculdade/TCC/Hologic_Over_GE_1"
    train_dataset = SemanticSegmentationDataset(dataset_folder,"training.csv", feature_extractor)
    val_dataset = SemanticSegmentationDataset(dataset_folder,"test.csv", feature_extractor)
    test_dataset = SemanticSegmentationDataset(dataset_folder,"validation.csv", feature_extractor)

    batch_size = 8
    num_workers = 2
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    segformer_finetuner = SegformerFinetuner(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        metrics_interval=10,
    )


    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=10,
        verbose=False,
        mode="min",
    )

    checkpoint_callback = ModelCheckpoint(save_top_k=1, monitor="val_loss")

    trainer = pl.Trainer(
        gpus=1,
        callbacks=[early_stop_callback, checkpoint_callback],
        max_epochs=int(os.environ["EPOCHS"]),
        val_check_interval=len(train_dataloader),
    )
    trainer.fit(segformer_finetuner)

    return trainer
