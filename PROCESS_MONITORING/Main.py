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
    #consider in model there is 5 steps which labelled an trained in model which named first,second,third,fourth,fifth
    # labellist contains a predicted labels with [score,xmin,ymin,xmax,ymax,class_name,cx,cy] features of that label
    for line in labellist:
        if line[5]=="first":
            return 1
        elif line[5]=="second":
            return 2
        elif line[5]=="third":
            return 3
        elif line[5]=="fourth":
            return 4
        elif line[5]=="fifth":
            return 5
        
    

def main():
    process_list=[]
    while(True):
        image=None
        image=cap_obj.frame
        if image is not None:
            if len(process_list)==5:
                if process_list==sorted(process_list):
                    print("Process Sequence Correct,starting new Process")
                    process_list=[]
                else:
                    print("Process Sequence Incorrect")
                    process_list=[]
            labellist,maskImage=mask_obj.run_inference(image)
            if len(labellist)>0:
                ProcessNumber=checkLabelList(labellist)
                
                if ProcessNumber is not None and ProcessNumber not in process_list:
                    if os.path.exists(INF_PATH) is False:
                        os.mkdir(INF_PATH)
                    fileName=os.path.join(INF_PATH,f"IMG_{int(time.time())*1000}.jpg")
                    cv2.imwrite(fileName,maskImage)
                    if os.path.exists(RAW_PATH) is False:
                        os.mkdir(RAW_PATH)
                    fileName=os.path.join(RAW_PATH,f"IMG_{int(time.time())*1000}.jpg")
                    cv2.imwrite(fileName,image)
                    process_list.append[ProcessNumber]
                

if __name__=="__main__":
    main()