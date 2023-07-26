from genericpath import exists
import os 
import shutil
import cv2
import xml
import time
from MASK_RCNN import maskRCNN
from IP_CAMERA_BUFFERELESS import BufferLessVideoCapture
import xml.etree.ElementTree as ET
################ varialble declaration and initialization  ###################
CODE_PATH=os.getcwd()
XML_CONFIG=os.path.join(CODE_PATH,"MAIN.xml")
root_node = ET.parse(XML_CONFIG).getroot()
MASK_LOG=os.path.join(CODE_PATH,root_node.find("LOG_PATH").text)
mask_obj=maskRCNN(CODE_PATH,XML_CONFIG,MASK_LOG)
cap_obj=BufferLessVideoCapture(XML_PATH=XML_CONFIG)
INF_PATH=os.path.join(CODE_PATH,root_node.find("LOG_PATH").text)
RAW_PATH=os.path.join(CODE_PATH,root_node.find("LOG_PATH").text)


def checkLabelList(labellist):
    # labellist contains a predicted labels with [score,xmin,ymin,xmax,ymax,class_name,cx,cy] features of that label
    ROI_X_MIN=root_node.find("ROI_X_MIN").text
    ROI_X_MAX=root_node.find("ROI_X_MAX").text
    ROI_Y_MIN=root_node.find("ROI_Y_MIN").text
    ROI_Y_MAX=root_node.find("ROI_Y_MAX").text
    # define a roi in which if hand is entered it will shows a warning
    hand_detected=[]
    for line in labellist:
        if line[5]=="hand":
            predicted_x_min=line[1]
            predicted_y_min=line[2]
            predicted_x_max=line[3]
            predicted_y_max=line[4]
            if ((predicted_x_min >= ROI_X_MIN and predicted_y_min >= ROI_Y_MIN) and
                (predicted_x_min <= ROI_X_MAX and predicted_y_min <= ROI_Y_MAX)):
                return True
            elif ((predicted_x_min >= ROI_X_MIN and predicted_y_max >= ROI_Y_MIN) and
                (predicted_x_min <= ROI_X_MAX and predicted_y_max <= ROI_Y_MAX)):
                return True
            elif ((predicted_x_max >= ROI_X_MIN and predicted_y_min >= ROI_Y_MIN) and
                (predicted_x_max <= ROI_X_MAX and predicted_y_min <= ROI_Y_MAX)):
                return True
            elif ((predicted_x_max >= ROI_X_MIN and predicted_y_max >= ROI_Y_MIN) and
                (predicted_x_max <= ROI_X_MAX and predicted_y_max <= ROI_Y_MAX)):
                return True
            else:
                return False
    return False
            

def main():
    while(True):
        image=None
        image=cap_obj.frame
        if image is not None:
            labellist,maskImage=mask_obj.run_inference(image)
            if len(labellist)>0:
                isHandDetectedInRoi=checkLabelList(labellist)
                if isHandDetectedInRoi is True:
                    print("Warning , Hand is Detected in ROI")
                    if os.path.exists(INF_PATH) is False:
                        os.mkdir(INF_PATH)
                    fileName=os.path.join(INF_PATH,f"IMG_{int(time.time())*1000}.jpg")
                    cv2.imwrite(fileName,maskImage)
                    if os.path.exists(RAW_PATH) is False:
                        os.mkdir(RAW_PATH)
                    fileName=os.path.join(RAW_PATH,f"IMG_{int(time.time())*1000}.jpg")
                    cv2.imwrite(fileName,image)
                else:
                    print("Everything is Normal")





if __name__=="__main__":
    main()