import numpy as np
import cv2
import matplotlib.pyplot as plt

#capture an image
cap = cv2.VideoCapture(0)
# to read image of google
#image = cv2.imread('image.jpg',-1) -1 equals to reading rgb
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#ret bool value 1 means cam is working 0 otherwise
faceData = [] 
#how many faces we gonna use
faceCount = 0
ret,  frame = cap.read()
grayFace = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
cap.release()
plt.imshow(frame)
plt.imshow(grayFace,cmap='gray')

faces = faceCascade.detectMultiScale(grayFace,1.5,5)#detect multiple faces set the point value to fit the frame

x,y,w,h = faces[0,:]#detect one face
#multiple face detection
#display number of person
names ={
        0:"Person 1",
        1:"Person 2",
        3:"Person 3"
        }
i = 0
for(x,y,w,h) in faces:
    output = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)#color yellow and thickness is 2
    name=names[i]
    cv2.putText(frame,name,(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255))#font size,color look at syntax
    i+=1
plt.imshow(cv2.cvtColor(output,cv2.COLOR_BGR2RGB))    

#croppedFace = frame[y:y+h,x:x+w]#get the cropped face



while True :
    ret, frame = cap.read()
    
    if ret == True:#image is captured
        pass
        grayFace = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)  #image is converted to gray
    else:
        print("Camera Error")
        
        
#write a program to load 20 cropped face iamges in face data
        
        