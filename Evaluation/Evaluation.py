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

from ..Util.Util import safe_make_folder
from ..Util import SQLogger

def create_vis_pred_img(img,nipple_mask,pectoral_mask,fibroglandular_tissue_mask,fatty_tissue_mask):
    pred_vis = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
    pred_vis[np.where(nipple_mask>0.5)] = [0,255,0]
    pred_vis[np.where(pectoral_mask>0.5)] = [0,0,255]
    pred_vis[np.where(fibroglandular_tissue_mask>0.5)] = [255,0,255]
    pred_vis[np.where(fatty_tissue_mask>0.5)] = [255,255,0]
    return pred_vis

def visual_evaluation(modelPath:str):
    import onnxruntime as ort

    # model = ort.InferenceSession('data/results/training/execution_2023_08_07_19_06_39_PM/model.onnx', providers=['CUDAExecutionProvider']) #loading model
    model = ort.InferenceSession(modelPath, providers=['CUDAExecutionProvider']) #loading model
    model_input_name = model.get_inputs()[0].name #getting input name for the model
    model.run(None, {model_input_name: np.zeros((1,384,384,1),dtype=np.float32)})

    execution_name = "execution_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%p")
    res_path = os.path.join(os.environ["EVALUTAION_FOLDER"],execution_name)
    pred_vis_path = os.path.join(res_path,'pred_vis')
    safe_make_folder(res_path)
    safe_make_folder(pred_vis_path)
    in_path = os.environ["DATASET_FOLDER"]
    for filename in os.listdir(in_path):
        if not (filename.endswith('.png')):
            continue
        full_filename =os.path.join(in_path,filename)
        img = io.imread(full_filename)
        h_original = img.shape[0]
        w_original = img.shape[1]
        img_original = img.copy()
        img = cv2.resize(img, (384,384), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32)/255.0
        batch = img.reshape((1,img.shape[0],img.shape[1],1))
        batch_pred = model.run(None, {model_input_name: batch})[0]
        nipple_mask = batch_pred[0,:,:,1]>0.5
        pectoral_mask = batch_pred[0,:,:,2]>0.5
        fibroglandular_tissue_mask = batch_pred[0,:,:,3]>0.5
        fatty_tissue_mask = batch_pred[0,:,:,4]>0.5
        nipple_mask = cv2.resize(nipple_mask.astype(np.uint8), (w_original,h_original), interpolation=cv2.INTER_NEAREST)
        pectoral_mask = cv2.resize(pectoral_mask.astype(np.uint8), (w_original,h_original), interpolation=cv2.INTER_NEAREST)
        fibroglandular_tissue_mask = cv2.resize(fibroglandular_tissue_mask.astype(np.uint8), (w_original,h_original), interpolation=cv2.INTER_NEAREST)
        fatty_tissue_mask = cv2.resize(fatty_tissue_mask.astype(np.uint8), (w_original,h_original), interpolation=cv2.INTER_NEAREST)
        out_img = create_vis_pred_img(img_original,nipple_mask,pectoral_mask,fibroglandular_tissue_mask,fatty_tissue_mask)
        out_filename = os.path.join(pred_vis_path,filename)
        io.imsave(out_filename,out_img)

def compute_iou(in_mask_gt, in_mask_pred):
    mask_gt = in_mask_gt > 0
    mask_pred = in_mask_pred > 0
    intersection = np.count_nonzero(mask_gt*mask_pred)
    union = np.count_nonzero((mask_gt+mask_pred)>0)
    if union < 1:
        return 1.0
    else:
        return intersection/union


def numerical_evaluation(modelPath:str,dataset_evaluation_folder=os.environ["EVALUATION_DATASET_FOLDER"]):
    import onnxruntime as ort
    logger = SQLogger.ExperimentLogger(os.environ["DBLOG_FOLDER"])

    model = ort.InferenceSession(modelPath, providers=['CUDAExecutionProvider']) #loading model
    model_input_name = model.get_inputs()[0].name #getting input name for the model
    model.run(None, {model_input_name: np.zeros((1,384,384,1),dtype=np.float32)})

    execution_name = "execution_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%p")
    res_path = os.path.join(os.environ["EVALUTAION_FOLDER"],execution_name)
    pred_vis_path = os.path.join(res_path,'pred_vis')
    gt_vis_path = os.path.join(res_path,'gt_vis')
    image_vis_path = os.path.join(res_path,'image_vis')
    full_vis_path = os.path.join(res_path,'full_vis')
    safe_make_folder(res_path)
    safe_make_folder(pred_vis_path)
    safe_make_folder(gt_vis_path)
    safe_make_folder(image_vis_path)
    safe_make_folder(full_vis_path)

    #
    # Start Evaluating 
    #
    
    dataset_folder = dataset_evaluation_folder
    #dataset_folder = "data/datasets/mlo-hologic2"
    image_folder=os.path.join(dataset_folder,"image")
    label_folder=os.path.join(dataset_folder,"label")
    target_csv_filename=os.path.join(dataset_folder,"validation.csv")
    raw_df = pd.read_csv(target_csv_filename)
    filenames = list(raw_df['filename'])

    iou_data = []
    iou_data_filename =  os.path.join(res_path,'iou.csv')

    for filename in filenames:
        full_filename =os.path.join(image_folder,filename)
        full_filename_gt =os.path.join(label_folder,filename)
        img = io.imread(full_filename)
        gt_labels = io.imread(full_filename_gt)
        h_original = img.shape[0]
        w_original = img.shape[1]
        img_original = img.copy()
        img = cv2.resize(img, (384,384), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32)/255.0
        batch = img.reshape((1,img.shape[0],img.shape[1],1))
        batch_pred = model.run(None, {model_input_name: batch})[0]
        nipple_mask = batch_pred[0,:,:,1]>0.5
        pectoral_mask = batch_pred[0,:,:,2]>0.5
        fibroglandular_tissue_mask = batch_pred[0,:,:,3]>0.5
        fatty_tissue_mask = batch_pred[0,:,:,4]>0.5
        nipple_mask = cv2.resize(nipple_mask.astype(np.uint8), (w_original,h_original), interpolation=cv2.INTER_NEAREST)
        pectoral_mask = cv2.resize(pectoral_mask.astype(np.uint8), (w_original,h_original), interpolation=cv2.INTER_NEAREST)
        fibroglandular_tissue_mask = cv2.resize(fibroglandular_tissue_mask.astype(np.uint8), (w_original,h_original), interpolation=cv2.INTER_NEAREST)
        fatty_tissue_mask = cv2.resize(fatty_tissue_mask.astype(np.uint8), (w_original,h_original), interpolation=cv2.INTER_NEAREST)
        
        nipple_mask_gt = gt_labels==1
        pectoral_mask_gt = gt_labels==2
        fibroglandular_tissue_mask_gt = gt_labels==3
        fatty_tissue_mask_gt = gt_labels==4

        nipple_iou = compute_iou(nipple_mask_gt,nipple_mask)
        pectoral_iou = compute_iou(pectoral_mask_gt,pectoral_mask)
        fibroglandular_tissue_iou = compute_iou(fibroglandular_tissue_mask_gt,fibroglandular_tissue_mask)
        fatty_tissue_iou = compute_iou(fatty_tissue_mask_gt,fatty_tissue_mask)
        logger.log_evaluation(modelPath.split('/')[-2], datetime.now(), os.path.basename(os.path.normpath(dataset_folder)), filename, nipple_iou, pectoral_iou, fibroglandular_tissue_iou, fatty_tissue_iou)
        
        iou_data.append([filename,nipple_iou,pectoral_iou,fibroglandular_tissue_iou,fatty_tissue_iou])

        pred_vis = create_vis_pred_img(img_original,nipple_mask,pectoral_mask,fibroglandular_tissue_mask,fatty_tissue_mask)
        pred_vis_filename = os.path.join(pred_vis_path,filename)
        io.imsave(pred_vis_filename,pred_vis)

        gt_vis = create_vis_pred_img(img_original,nipple_mask_gt,pectoral_mask_gt,fibroglandular_tissue_mask_gt,fatty_tissue_mask_gt)
        gt_vis_filename = os.path.join(gt_vis_path,filename)
        io.imsave(gt_vis_filename,gt_vis)

        img_vis_filename = os.path.join(image_vis_path,filename)
        io.imsave(img_vis_filename,img_original)

        full_vis = cv2.hconcat([cv2.cvtColor(img_original,cv2.COLOR_GRAY2RGB), gt_vis, pred_vis])
        full_vis_filename = os.path.join(full_vis_path,filename)
        io.imsave(full_vis_filename,full_vis)
    
    logger.close()
    df = pd.DataFrame(iou_data, columns =['Filename', 'Nipple IoU', 'Pectoral IoU', 'Fibr. tissue IoU', 'Fatty tissue IoU'])
    df.to_csv(iou_data_filename)

if __name__ == '__main__':
    #visual_evaluation()
    numerical_evaluation()


#
# All in one class evaluator
#
class Evaluator:
    def init(self,modelPath:str):
        self.modelPath = modelPath
        
        #Load the model
        import onnxruntime as ort
        self.model = ort.InferenceSession(modelPath, providers=['CUDAExecutionProvider']) #loading model
        self.model_input_name = self.model.get_inputs()[0].name #getting input name for the model
        self.model.run(None, {self.model_input_name: np.zeros((1,384,384,1),dtype=np.float32)})

    def evaluate(self,dataset_evaluation_folder:str = os.environ["EVALUATION_DATASET_FOLDER"]):
        execution_name = "execution_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%p")
        res_path = os.path.join(os.environ["EVALUTAION_FOLDER"],execution_name)

        dataset_folder = dataset_evaluation_folder 
        image_folder=os.path.join(dataset_folder,"image")
        label_folder=os.path.join(dataset_folder,"label")
        target_csv_filename=os.path.join(dataset_folder,"validation.csv")
        raw_df = pd.read_csv(target_csv_filename)

        # Create folders
        execution_name = "execution_"+datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%p")
        res_path = os.path.join(os.environ["EVALUTAION_FOLDER"],execution_name)
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

            #
            # Precison
            #
            precision_nipple = self.compute_precision(nipple_mask_gt, nipple_mask)
            precision_pectoral = self.compute_precision(pectoral_mask_gt, pectoral_mask)
            precision_fibroglandular_tissue = self.compute_precision(fibroglandular_tissue_mask_gt, fibroglandular_tissue_mask)
            precision_fatty_tissue = self.compute_precision(fatty_tissue_mask_gt, fatty_tissue_mask)


            #
            # Intersection Over Union
            #
            nipple_iou = self.compute_iou(nipple_mask_gt, nipple_mask)
            pectoral_iou = self.compute_iou(pectoral_mask_gt, pectoral_mask)
            fibroglandular_tissue_iou = self.compute_iou(fibroglandular_tissue_mask_gt, fibroglandular_tissue_mask)
            fatty_tissue_iou = self.compute_iou(fatty_tissue_mask_gt, fatty_tissue_mask)

             # Append Data
            evaluation_data.append([filename, 
                                    nipple_iou, pectoral_iou, fibroglandular_tissue_iou, fatty_tissue_iou,
                                    precision_nipple, precision_pectoral, precision_fibroglandular_tissue, precision_fatty_tissue])

        # Create DataFrame and save to CSV
        columns = ['filename', 'nipple_iou', 'pectoral_iou', 'fibroglandular_tissue_iou', 'fatty_tissue_iou',
                   'precision_nipple', 'precision_pectoral', 'precision_fibroglandular_tissue', 'precision_fatty_tissue']


        result_df = pd.DataFrame(evaluation_data, columns=columns)
        result_df.to_csv(os.path.join(res_path, execution_name + "_evaluation.csv"), index=False)

        return result_df

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


