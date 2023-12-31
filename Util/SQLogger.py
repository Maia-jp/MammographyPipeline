import sqlite3
import datetime

class ExperimentLogger:
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()

    def create_experiment(self, id, architecture, dataset):
        created_at = datetime.datetime.now()
        self.cursor.execute("INSERT INTO Experiment (id, created_at, architecture, dataset) VALUES (?, ?, ?, ?)",
                            (id, created_at, architecture, dataset))
        self.conn.commit()

    def log_evaluation(self, experiment_id, created_at, dataset, filename, nipple_iou, pectoral_iou, fibroglandular_tissue_iou, fatty_tissue_iou):                            
           self.cursor.execute("INSERT INTO Evaluation (id, created_at, dataset, 'Filename', 'Nipple IoU', 'Pectoral IoU', 'Fibr. tissue IoU', 'Fatty tissue IoU') VALUES (?, ?, ?, ?, ?, ?, ?, ?)",                                                                                                                                                           
                               (experiment_id, created_at, dataset, filename, nipple_iou, pectoral_iou, fibroglandular_tissue_iou, fatty_tissue_iou))                            
           self.conn.commit()   

    def log_history(self, experiment_id, epoch, iou_score, loss, val_iou_score, val_loss):
        self.cursor.execute("INSERT INTO History (experiment, epoch, iou_score, loss, val_iou_score, val_loss) VALUES (?, ?, ?, ?, ?, ?)",
                            (experiment_id, epoch, iou_score, loss, val_iou_score, val_loss))
        self.conn.commit()

    def close(self):
        self.conn.close()