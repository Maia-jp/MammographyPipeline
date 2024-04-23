import pytorch_lightning as pl
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from datasets import load_metric
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
# import os
from PIL import Image
import numpy as np
import random
import pandas as pd

class SegformerFinetuner(pl.LightningModule):

    def __init__(self, train_dataloader=None, val_dataloader=None, test_dataloader=None, metrics_interval=100):
        super(SegformerFinetuner, self).__init__()
        self.num_classes = 4
        self.metrics_interval = metrics_interval
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.test_dl = test_dataloader

        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b0-finetuned-ade-512-512",
            return_dict=False,
            ignore_mismatched_sizes=True,
        )

        self.train_mean_iou = load_metric("mean_iou")
        self.val_mean_iou = load_metric("mean_iou")
        self.test_mean_iou = load_metric("mean_iou")

    def forward(self, images, masks):
        outputs = self.model(pixel_values=images, labels=masks)
        return(outputs)

    def training_step(self, batch, batch_nb):

      images, masks = batch['pixel_values'], batch['labels']

      outputs = self(images, masks)

      loss, logits = outputs[0], outputs[1]

      upsampled_logits = nn.functional.interpolate(
          logits,
          size=masks.shape[-2:],
          mode="bilinear",
          align_corners=False
      )

      predicted = upsampled_logits.argmax(dim=1)

      self.train_mean_iou.add_batch(
          predictions=predicted.detach().cpu().numpy(),
          references=masks.detach().cpu().numpy()
      )
      if batch_nb % self.metrics_interval == 0:

          metrics = self.train_mean_iou.compute(
              num_labels=self.num_classes,
              ignore_index=255,
              reduce_labels=False,
          )

          metrics = {'loss': loss, "mean_iou": metrics["mean_iou"], "mean_accuracy": metrics["mean_accuracy"]}

          for k,v in metrics.items():
              self.log(k,v)

          return(metrics)
      else:
          return({'loss': loss})

    def validation_step(self, batch, batch_nb):

      images, masks = batch['pixel_values'], batch['labels']

      outputs = self(images, masks)

      loss, logits = outputs[0], outputs[1]

      upsampled_logits = nn.functional.interpolate(
          logits,
          size=masks.shape[-2:],
          mode="bilinear",
          align_corners=False
      )

      predicted = upsampled_logits.argmax(dim=1)

      self.val_mean_iou.add_batch(
          predictions=predicted.detach().cpu().numpy(),
          references=masks.detach().cpu().numpy()
      )

      return({'val_loss': loss})

    def validation_epoch_end(self, outputs):
      metrics = self.val_mean_iou.compute(
            num_labels=self.num_classes,
            ignore_index=255,
            reduce_labels=False,
        )

      avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
      val_mean_iou = metrics["mean_iou"]
      val_mean_accuracy = metrics["mean_accuracy"]

      metrics = {"val_loss": avg_val_loss, "val_mean_iou":val_mean_iou, "val_mean_accuracy":val_mean_accuracy}
      for k,v in metrics.items():
          self.log(k,v)

      return metrics

    def test_step(self, batch, batch_nb):

      images, masks = batch['pixel_values'], batch['labels']

      outputs = self(images, masks)

      loss, logits = outputs[0], outputs[1]

      upsampled_logits = nn.functional.interpolate(
          logits,
          size=masks.shape[-2:],
          mode="bilinear",
          align_corners=False
      )

      predicted = upsampled_logits.argmax(dim=1)

      self.test_mean_iou.add_batch(
          predictions=predicted.detach().cpu().numpy(),
          references=masks.detach().cpu().numpy()
      )

      return({'test_loss': loss})

    def test_epoch_end(self, outputs):
      metrics = self.test_mean_iou.compute(
            num_labels=self.num_classes,
            ignore_index=255,
            reduce_labels=False,
        )

      avg_test_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
      test_mean_iou = metrics["mean_iou"]
      test_mean_accuracy = metrics["mean_accuracy"]

      metrics = {"test_loss": avg_test_loss, "test_mean_iou":test_mean_iou, "test_mean_accuracy":test_mean_accuracy}

      for k,v in metrics.items():
          self.log(k,v)

      return metrics

    def configure_optimizers(self):
      return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)

    def train_dataloader(self):
      return self.train_dl

    def val_dataloader(self):
      return self.val_dl

    def test_dataloader(self):
      return self.test_dl
