from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *     #from sort import everything



cap=cv2.VideoCapture(r"C:\Users\siddh\proj\dataset\traffic.mp4")  #for video
model=YOLO(r"C:\Users\siddh\proj\yolo weights\yolov8m.pt")    #yolo version 8 medium


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


mask=cv2.imread(r"C:\Users\siddh\proj\dataset\mask1.png")    #reading mask
mask = cv2.resize(mask, (1280, 720))   #resizing mask


#tracking
tracker=Sort(max_age=750,min_hits=10,iou_threshold=0.4)

totalcount=[]


# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_time = int(1000 / fps)  # Calculate delay in milliseconds



while True:
    success,img=cap.read()    #reading video
    
    
    #resizing image
    img = cv2.resize(img, (1280, 720))
    
    # Apply the mask to the image   #NOTE:img and mask should be of same size
    imgRegion = cv2.bitwise_and(img, mask)

    result=model(imgRegion,stream=True)      #integrating yolo
      
    detections=np.empty((0,5))
    


    for r in result:            #r is each frame
        boxes=r.boxes           #each frame has a box which will be given by r.boxes
        for box in boxes:
            x1,y1,x2,y2=box.xyxy[0]                     #bounding box
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            print(x1,y1,x2,y2) 

            #confidence values
             
            conf=math.ceil((box.conf[0]*100))/100                      #confidence values and rounding them to 2 decimal places
            
             
            #classes
            cls=int(box.cls[0])
            currentClass=classNames[cls]

            if(currentClass=="car" or currentClass=="truck" or currentClass=="bus" or currentClass=="motorbike"):          #displaying for a certain class
             #cvzone.putTextRect(img,f'{currentClass} {conf}',(max(0,x1),max(35,y1-5)),scale=0.8,thickness=1,offset=1)
             #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),2,4)
             currentArray=np.array([x1,y1,x2,y2,conf])
             detections=np.vstack((detections,currentArray))

    resultsTracker=tracker.update(detections)


    #making line
    cv2.line(img,(700,400),(1100,400),(0,255,0),4,3)

    for result in resultsTracker:
       x1,y1,x2,y2,id=result 
       x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)

       w,h=x2-x1,y2-y1                                         #calculating width and height

       cvzone.putTextRect(img,f'id: {id}',(max(0,x1),max(35,y1-5)),scale=1,thickness=2,offset=1)
       cv2.rectangle(img,(x1,y1),(x2,y2),(0,50,255),2,4)

       cx,cy=x1+w//2,y1+h//2
       cv2.circle(img,(cx,cy),5,(255,219,79),cv2.FILLED)

       if (700<cx<1100 and 375<cy<425):
          if(totalcount.count(id)==0):                                          #checking whether this id is there in list  
           totalcount.append(id)
    cvzone.putTextRect(img,f'Count:{len(totalcount)}',(50,100),thickness=3)
          
          

    cv2.imshow("clip", img)
    
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break  # Exit loop when 'q' is pressed


    

    
cap.release()    #to release videolol
cv2.destroyAllWindows()  #close all windows           