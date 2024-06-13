from ultralytics import YOLO
import cv2
import cvzone
import math

#cap=cv2.VideoCapture(0) #for webcam
#cap.set(3,1280)                                                          #setting width(id=3)
#cap.set(4,720)                                                           #setting height


cap=cv2.VideoCapture(0)  #for webcam

model=YOLO(r"C:\Users\siddh\proj\yolo weights\yolov8n.pt")


classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]                                                                                               #list of classes
    

while True:
    success,img=cap.read()
    img=cv2.resize(img,(1080,720))
    

    result=model(img,stream=True) #integrating yolo
    




    for r in result:            #r is each frame
        boxes=r.boxes           #each frame has a box which will be given by r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]                     #bounding box
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            print(x1,y1,x2,y2)
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2,4) 

            #confidence values
             
            conf=math.ceil((box.conf[0]*100))/100                      #confidence values and rounding them to 2 decimal places
            cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(35,y1-5)),scale=1,thickness=2) #putting text on rectangle. max(0,...) says agar values negative me jaa rha hai toh take 0 for x and 35 for y.

             
            #classes
            cls=int(box.cls[0])
            cvzone.putTextRect(img,f'{classNames[cls]} {conf}',(max(0,x1),max(35,y1-5)),scale=1,thickness=2 )



    cv2.imshow("Webcam", img)
    
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break  # Exit loop when 'q' is pressed

    
cap.release()
cv2.destroyAllWindows() 