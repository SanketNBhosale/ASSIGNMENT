import cv2
import os
import queue
import threading
import xml
import xml.etree.ElementTree as ET
class BufferLessVideoCapture:
    def __init__(self,XML_PATH):
        self.moduleStop=False
        self.frame=None
        self.CODE_PATH=os.getcwd()
        XML_CONFIG=os.path.join(self.CODE_PATH,"MAIN.xml")
        root_node = ET.parse(XML_CONFIG).getroot()
        self.USERNAME = root_node.find("IP_CAMERA_USERNAME").text
        self.PASSWORD = root_node.find("IP_CAMERA_PASSWORD").text
        self.CAM_IP=root_node.find("IP_CAMERA_IP_ADDRESS").text
        
        self.initCam()
        t = threading.Thread(target=self.reader)
        t.start()

    def initCam(self):
        self.cap = cv2.VideoCapture(f"rtsp://{self.USERNAME}:{self.PASSWORD}@{self.CAM_IP}/Streaming/Channels/1")

    def reader(self):
        while not self.moduleStop:
            try: 
                if not self.cap.isOpened():
                    self.initCam()  
                ret, frame = self.cap.read()
                if ret:
                    self.frame=frame
            except Exception as E:
                print("FRAME CAPTURE EXCEPTION ",E)
                self.initCam()
                pass            
   


