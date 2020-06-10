# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 15:58:18 2019

@author: Shrikrishna Warkhedi
"""

import numpy as np
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX

########################################### FOR 2.1 -2.4 ###########################################

filename = 'D:\\Sem 2\\CV\\CV2.mp4'

video = cv2.VideoCapture(filename)
frame_width = int(video.get(3))
frame_height = int(video.get(4))
outvid = cv2.VideoWriter('D:\\Sem 2\\CV\\part1_1.mp4',cv2.VideoWriter_fourcc('D','I','V','X'), 30, (frame_width,frame_height))
frames_counter = 1
fps = int(video.get(cv2.CAP_PROP_FPS))

frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
ksize =1
cpar1 = 25
cpar2 = 50
while True:
    check, frame = video.read()
    frames_counter = frames_counter + 1
    videotime = int(frames_counter / fps)
    if check:
        
            if videotime < 5:

                #cv2.imshow("Color",frame)
                outvid.write(frame)
            if videotime > 4 and videotime < 10:
                output = frame.copy()
                
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (7,7), sigmaX=-1, sigmaY=-1)
  
                cimg = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
                
                #print("VideoTime:",videotime)
                #outvid.write(np.uint8(sobel_horizontal))
                if videotime%2 == 1:
                     sobel_horizontal = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=1) 
                     finimg = sobel_horizontal
                     msg = "EdgeDetection: Sobel_Horizontal, kernelSize: "+str(1)
                     cv2.putText(finimg,msg,(50,50), font, 1,(56,134,0),5,cv2.LINE_AA)
                     outvid.write(np.uint8(finimg))
                if videotime%2 == 0:
                     sobel_vertical = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)
                     finimg = sobel_vertical
                     msg = "EdgeDetection: Sobel_Vertical, kernelSize: "+str(5)
                     cv2.putText(finimg,msg,(50,50), font, 1,(0,255,255),5,cv2.LINE_AA)
                     outvid.write(np.uint8(finimg))
                     
            if videotime > 10 and videotime < 16:
                
                if videotime % 2 == 1:
                    cpar1 = 2+cpar1
                    cpar2 = 2+cpar2
                else :
                    cpar1 = cpar1 - 2
                    cpar2 = cpar1 - 2
                    
                cannyEdges = cv2.Canny(frame,cpar1,cpar2)
                msg = "Canny Edges. Parameters: "+str(cpar1)+", "+str(cpar2) 
                cv2.putText(cannyEdges,msg,(50,50), font, 1,(255,255,0),1,cv2.LINE_AA)
                outvid.write(cv2.cvtColor(cannyEdges, cv2.COLOR_GRAY2BGR))
            if videotime > 16:
                break
            
            
      
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

print("Number of frames in the video: ", frames_counter)
video.release()
outvid.release()
cv2.destroyAllWindows() 

######################################## FOR 2.5 ########################################

filename = 'D:\\Sem 2\\CV\\circleVid2.mp4'
video = cv2.VideoCapture(filename)
frame_width = int(video.get(3))
frame_height = int(video.get(4))
outvid = cv2.VideoWriter('D:\\Sem 2\\CV\\part2_2.mp4',cv2.VideoWriter_fourcc('D','I','V','X'), 17, (frame_width,frame_height))
frames_counter = 1
fps = int(video.get(cv2.CAP_PROP_FPS))+1

frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

mindist = 30
param1 = 100
param2 = 50
while True:
    check, frame = video.read()
    frames_counter = frames_counter + 1
    videotime = int(frames_counter / fps)
    print("Videotime",videotime)
    if check:
        output = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7,7), sigmaX=-1, sigmaY=-1)  
        if videotime == 7:
            param1 =90
            param2 =40
            mindist =35
            
        print(param1,param2)
        msg = "Hough Circle parameters: minDist= "+str(mindist)+" param1= "+str(param1)+" param2= "+str(param2)
        
        circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,
                    1,mindist, param1=param1,param2=param2,minRadius=0,maxRadius=0)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(output, (x - 150, y - 150), (x + 150, y + 150), (0, 128, 255), 5)
        cv2.putText(output,msg,(50,50), font, 1,(255,255,0),1,cv2.LINE_AA)  
        cv2.imshow('detected circles',output)
        outvid.write(output)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
video.release()
outvid.release()
cv2.destroyAllWindows() 


######################################## FOR 2.5 onwards ###########################################

MIN_MATCH_COUNT=15

detector=cv2.xfeatures2d.SIFT_create()

FLANN_INDEX_KDITREE=0
flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})

trainImg=cv2.imread('D:\\Sem 2\\CV\\haldiTrain.jpg',0)
trainKP,trainDesc=detector.detectAndCompute(trainImg,None)

filename = 'D:\\Sem 2\\CV\\haldiCut.mp4'

video = cv2.VideoCapture(filename)
frame_width = int(video.get(3))
frame_height = int(video.get(4))
outvid = cv2.VideoWriter('D:\\Sem 2\\CV\\part4.mp4',cv2.VideoWriter_fourcc('D','I','V','X'), 17, (frame_width,frame_height))
frames_counter = 1
fps = int(video.get(cv2.CAP_PROP_FPS))+1
frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
font = cv2.FONT_HERSHEY_SIMPLEX


while True:
    ret, QueryImgBGR=video.read()
    QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)
    queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)
    matches=flann.knnMatch(queryDesc,trainDesc,k=2)

    goodMatch=[]
    for m,n in matches:
        if(m.distance<0.75*n.distance):
            goodMatch.append(m)
    if(len(goodMatch)>MIN_MATCH_COUNT):
        tp=[]
        qp=[]
        for m in goodMatch:
            tp.append(trainKP[m.trainIdx].pt)
            qp.append(queryKP[m.queryIdx].pt)
        tp,qp=np.float32((tp,qp))
        H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
        if H is not None:
            h,w=trainImg.shape
            trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
            queryBorder=cv2.perspectiveTransform(trainBorder,H)
            cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,255,0),5)
    else:
        msg = "Minimum KNN Match not met! %d/%d"%(len(goodMatch),MIN_MATCH_COUNT)
        cv2.putText(QueryImgBGR,msg,(50,50), font, 1,(255,255,0),1,cv2.LINE_AA)
        print ("Not Enough match found- %d/%d"%(len(goodMatch),MIN_MATCH_COUNT))
     #cv2.imshow('result',QueryImgBGR)
    outvid.write(QueryImgBGR)
    if cv2.waitKey(1) | 0xFF==ord('q'):
        break
    
video.release()
outvid.release()
cv2.destroyAllWindows()
