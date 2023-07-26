from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog,DatasetCatalog
from detectron2.structures import BoxMode
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import ColorMode
import cv2
import json
import logging
import xml.etree.ElementTree as ET
import os

class maskRCNN:
    
    def __init__(self,CODE_PATH,CONFIG_FILE,LOG_FILENAME):
        root_node = ET.parse(CONFIG_FILE).getroot()
        modelJsonPath=os.path.join(CODE_PATH,root_node.find("MODEL_PATH").text,root_node.find("JSON_PATH").text)
        json_file = open(modelJsonPath)
        data = json.load(json_file) 
        self.all_class_name = list(data.values())
        self.numclasses = len(self.all_class_name)
        self.predictor=None
        self.mrcnn_config_fl=root_node.find("CONFIG_PATH").text
        self.mrcnn_model_loc= root_node.find("MODEL_PATH").text
        self.mrcnn_model_fl= root_node.find("MODEL_FILENAME").text           
        self.detection_thresh= root_node.find("THRESHHOLD").text
        self.register_modeldatasets()
        self.logger = None
        log_format=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger=logging.getLogger(LOG_FILENAME)
        self.logger.setLevel(logging.DEBUG)
        logger_fh=logging.FileHandler(LOG_FILENAME,mode='a')
        logger_fh.setFormatter(log_format)
        logger_fh.setLevel(logging.DEBUG)
        self.logger.addHandler(logger_fh)

    def register_modeldatasets(self):
        
        tag_name="test"
        MetadataCatalog.get(tag_name).set(thing_classes=self.ALL_CLASS_NAMES)
        self.mahindra_pdi_metadata = MetadataCatalog.get(tag_name)
        cfg = get_cfg()
        cfg.merge_from_file(self.mrcnn_config_fl)
        #cfg.MODEL.DEVICE='cpu'   # COMMENT IF CUDA ENABLED GPU AVAILABLE
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = self.NUMCLASSES 
        cfg.OUTPUT_DIR=self.mrcnn_model_loc
        cfg.MODEL.WEIGHTS =os.path.join(cfg.OUTPUT_DIR,self.mrcnn_model_fl)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.detection_thresh
        self.predictor = DefaultPredictor(cfg)

    def run_inference(self,img):
        global logger
        allLabelNameList = []
        output = self.predictor(img)
        predictions=output["instances"].to("cpu")
        boxes_surface = predictions.pred_boxes.tensor.to("cpu").numpy()
        pred_class_surface = predictions.pred_classes.to("cpu").numpy()
        scores_surface = predictions.scores.to("cpu").numpy() 
        for i,box in enumerate(boxes_surface):
            class_name = self.mahindra_pdi_metadata.get("thing_classes")[pred_class_surface[i]]
            score = scores_surface[i]
            box = boxes_surface[i]
            ymin = int(box[1])
            xmin = int(box[0])
            ymax = int(box[3])
            xmax = int(box[2])
            labellistsmall = []
            labellistsmall.append(score)
            labellistsmall.append(xmin)
            labellistsmall.append(ymin)
            labellistsmall.append(xmax)
            labellistsmall.append(ymax)
            labellistsmall.append(class_name)
            cx,cy = self.get_centroid(xmin, xmax, ymin, ymax)
            labellistsmall.append(cx)
            labellistsmall.append(cy)
            allLabelNameList.append(labellistsmall)
            self.drawCV2Box(img,class_name,xmin,ymin,xmax,ymax, False)
        return allLabelNameList, img

    
    def drawCV2Box(self,frame,labelName, xmin,ymin,xmax,ymax, isdefect):
        try:
            cv2.putText(frame, labelName, (xmin,ymin-15), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (255, 255, 255), 2, cv2.LINE_AA)  
            if isdefect is True:                         
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0,0,255),3)
            else:
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (0,255,0),3)
                
        except Exception as e:
            print("Exception in drawCV2Box() : "+ str(e))
 
    def get_centroid(self,xmin, xmax, ymin, ymax):
        cx = int((xmin + xmax) / 2.0)
        cy = int((ymin + ymax) / 2.0)
        return cx, cy 

