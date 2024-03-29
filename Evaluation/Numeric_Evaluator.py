import os
import shutil
import cv2
import re
import pandas as pd
import numpy as np
import random
from skimage import img_as_uint, img_as_float, exposure, draw
import skimage.io as io
from datetime import datetime
from progressbar import ProgressBar, Percentage, GranularBar, \
    Timer, ETA, Counter

from ..Util.Util import safe_make_folder
from ..Util import SQLogger
from ..Evaluation import Evaluator


class Numeric_Evaluator:

    def __init__(self,model_path:str):
        self.modelPath = model_path

        import onnxruntime as ort
        self.model = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider']) #loading model
        self.model_input_name = self.model.get_inputs()[0].name #getting input name for the model
        self.model.run(None, {self.model_input_name: np.zeros((1,384,384,1),dtype=np.float32)})

    def evaluate(self,evaluation_folder:str = os.environ["EVALUTAION_FOLDER"],dataset_evaluation_folder:str = os.environ["EVALUATION_DATASET_FOLDER"]) -> pd.DataFrame:
        execution_name = "execution_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%p")
        res_path = os.path.join(evaluation_folder,execution_name)

        dataset_folder = dataset_evaluation_folder 
        image_folder=os.path.join(dataset_folder,"image")
        label_folder=os.path.join(dataset_folder,"label")
        target_csv_filename=os.path.join(dataset_folder,"validation.csv")
        raw_df = pd.read_csv(target_csv_filename)

        # Create folders
        execution_name = "execution_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%p")
        res_path = os.path.join(evaluation_folder,execution_name)
        pred_vis_path = os.path.join(res_path,'pred_vis')
        gt_vis_path = os.path.join(res_path,'gt_vis')
        image_vis_path = os.path.join(res_path,'image_vis')
        full_vis_path = os.path.join(res_path,'full_vis')
        safe_make_folder(res_path)
        safe_make_folder(pred_vis_path)
        safe_make_folder(gt_vis_path)
        safe_make_folder(image_vis_path)
        safe_make_folder(full_vis_path)
        
        filenames = list(raw_df['filename'])

        evaluation_data = []

        with ProgressBar(max_value=len(filenames),widgets=[Percentage(), " ", GranularBar(), " ", Timer(), ]) as bar:
            i = 0
            for filename in filenames:
                #Load Images
                full_filename = os.path.join(image_folder, filename)
                full_filename_gt = os.path.join(label_folder, filename)
                img = io.imread(full_filename)
                gt_labels = io.imread(full_filename_gt)

                #Image Preprocessing
                h_original = img.shape[0]
                w_original = img.shape[1]
                img_original = img.copy()
                img = cv2.resize(img, (384, 384), interpolation=cv2.INTER_NEAREST)
                img = img.astype(np.float32) / 255.0
                batch = img.reshape((1, img.shape[0], img.shape[1], 1))

                #Model Prediction
                batch_pred = self.model.run(None, {self.model_input_name: batch})[0]

                #Tresholding Mask Generation
                nipple_mask = batch_pred[0, :, :, 1] > 0.5
                pectoral_mask = batch_pred[0, :, :, 2] > 0.5
                fibroglandular_tissue_mask = batch_pred[0, :, :, 3] > 0.5
                fatty_tissue_mask = batch_pred[0, :, :, 4] > 0.5


                nipple_mask_gt = gt_labels==1
                pectoral_mask_gt = gt_labels==2
                fibroglandular_tissue_mask_gt = gt_labels==3
                fatty_tissue_mask_gt = gt_labels==4

                # -- Compute --

                # Precison
                precision_nipple = self.compute_precision(nipple_mask_gt, nipple_mask)
                precision_pectoral = self.compute_precision(pectoral_mask_gt, pectoral_mask)
                precision_fibroglandular_tissue = self.compute_precision(fibroglandular_tissue_mask_gt, fibroglandular_tissue_mask)
                precision_fatty_tissue = self.compute_precision(fatty_tissue_mask_gt, fatty_tissue_mask)

                # Intersection Over Union
                nipple_iou = self.compute_iou(nipple_mask_gt, nipple_mask)
                pectoral_iou = self.compute_iou(pectoral_mask_gt, pectoral_mask)
                fibroglandular_tissue_iou = self.compute_iou(fibroglandular_tissue_mask_gt, fibroglandular_tissue_mask)
                fatty_tissue_iou = self.compute_iou(fatty_tissue_mask_gt, fatty_tissue_mask)


                # Accuracy
                accuracy_nipple = self.compute_accuracy(nipple_mask_gt, nipple_mask)
                accuracy_pectoral = self.compute_accuracy(pectoral_mask_gt, pectoral_mask)
                accuracy_fibroglandular_tissue = self.compute_accuracy(fibroglandular_tissue_mask_gt, fibroglandular_tissue_mask)
                accuracy_fatty_tissue = self.compute_accuracy(fatty_tissue_mask_gt, fatty_tissue_mask)

                # Dice Coefficient
                dice_nipple = self.compute_dice_coeff(nipple_mask_gt, nipple_mask)
                dice_pectoral = self.compute_dice_coeff(pectoral_mask_gt, pectoral_mask)
                dice_fibroglandular_tissue = self.compute_dice_coeff(fibroglandular_tissue_mask_gt, fibroglandular_tissue_mask)
                dice_fatty_tissue = self.compute_dice_coeff(fatty_tissue_mask_gt, fatty_tissue_mask)

                # Hausdorff Distance
                # hausdorff_nipple = self.compute_Hausdorff_distance(nipple_mask_gt, nipple_mask)
                # hausdorff_pectoral = self.compute_Hausdorff_distance(pectoral_mask_gt, pectoral_mask)
                # hausdorff_fibroglandular_tissue = self.compute_Hausdorff_distance(fibroglandular_tissue_mask_gt, fibroglandular_tissue_mask)
                # hausdorff_fatty_tissue = self.compute_Hausdorff_distance(fatty_tissue_mask_gt, fatty_tissue_mask)

                # Append Data
                evaluation_data.append([filename, 
                                        nipple_iou, pectoral_iou, fibroglandular_tissue_iou, fatty_tissue_iou,
                                        precision_nipple, precision_pectoral, precision_fibroglandular_tissue, precision_fatty_tissue,
                                        accuracy_nipple,accuracy_pectoral,accuracy_fibroglandular_tissue,accuracy_fatty_tissue,
                                        dice_nipple,dice_pectoral,dice_fibroglandular_tissue,dice_fatty_tissue])
                i += 1
                bar.update(i)

        # Create DataFrame and save to CSV
        columns = ['filename',
                    'nipple_iou', 'pectoral_iou', 'fibroglandular_tissue_iou', 'fatty_tissue_iou',
                    'precision_nipple', 'precision_pectoral', 'precision_fibroglandular_tissue', 'precision_fatty_tissue',
                    'accuracy_nipple', 'accuracy_pectoral', 'accuracy_fibroglandular_tissue', 'accuracy_fatty_tissue',
                    'dice_nipple', 'dice_pectoral', 'dice_fibroglandular_tissue', 'dice_fatty_tissue']



        result_df = pd.DataFrame(evaluation_data, columns=columns)
        result_df.to_csv(os.path.join(res_path, execution_name + "_evaluation.csv"), index=False)

        return result_df

    #
    # Numeric Evaluator Functions
    #
    def compute_iou(self,in_mask_gt, in_mask_pred):
        mask_gt = in_mask_gt > 0
        mask_pred = in_mask_pred > 0
        intersection = np.count_nonzero(mask_gt*mask_pred)
        union = np.count_nonzero((mask_gt+mask_pred)>0)
        if union < 1:
            return 1.0
        else:
            return intersection/union
        
    def compute_precision(self, gt_mask, pred_mask):
        true_positives = np.sum(np.logical_and(gt_mask, pred_mask))
        false_positives = np.sum(np.logical_and(np.logical_not(gt_mask), pred_mask))
        precision = true_positives / (true_positives + false_positives + 1e-9)  # to avoid division by zero
        return precision
    
    def compute_recall(self, gt_mask, pred_mask):
        num_true_positives = np.sum(np.logical_and(gt_mask, pred_mask))
        num_false_negatives = np.sum(np.logical_and(gt_mask, np.logical_not(pred_mask)))

        recall = num_true_positives / (num_true_positives + num_false_negatives + np.finfo(np.float32).eps)
        return recall
    
    def compute_accuracy(self, gt_mask, pred_mask):
        correct_pixels = np.sum(np.equal(gt_mask, pred_mask))
        total_pixels = gt_mask.size
        accuracy = correct_pixels / total_pixels
        return accuracy

    def compute_dice_coeff(self, gt_mask, pred_mask):
        intersection = np.sum(np.logical_and(gt_mask, pred_mask))
        union = np.sum(np.logical_or(gt_mask, pred_mask))
        dice_coefficient = (2.0 * intersection) / (union + 1e-9)  # to avoid division by zero
        return dice_coefficient