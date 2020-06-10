# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 15:58:18 2019

@author: Shrikrishna Warkhedi
"""

import numpy as np
import datetime
import cv2

filename = 'D:\\Sem 2\\CV\\CV2.mp4'

video = cv2.VideoCapture(filename)
frame_width = int(video.get(3))
frame_height = int(video.get(4))
outvid = cv2.VideoWriter('D:\\Sem 2\\CV\\morph.mp4',cv2.VideoWriter_fourcc('D','I','V','X'), 30, (frame_width,frame_height))
frames_counter = 1
background=0

fps = int(video.get(cv2.CAP_PROP_FPS))
for i in range(30):
	background = video.read()

frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

while True:
    check, frame = video.read()
    frames_counter = frames_counter + 1
    videotime = int(frames_counter / fps)
    if check:
        if videotime < 16:
            
            if videotime < 6:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray1 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                #cv2.imshow("Green",gray)
                #outvid.write(gray1)
            if videotime > 4 and videotime < 11:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray,(9,9),0)
                blur = cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR)
                #outvid.write(blur)
            if videotime > 10 and videotime < 16:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur2 = cv2.bilateralFilter(gray,10,90,90)  
                blur2 = cv2.cvtColor(blur2, cv2.COLOR_GRAY2BGR)
                #outvid.write(blur2)      
        else:
            
            if videotime > 14 and videotime < 20:
                outvid.write(frame)
            if videotime > 19 and videotime < 26:
                img = frame
                (B, G, R) = cv2.split(img)
                
                
                #thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
                #mask =cv2.threshold(img_rgb, 0, 255,cv2.THRESH_BINARY_INV |cv2.THRESH_OTSU)
                #img[mask == 255] = [0, 0, 255]
                R[R>0]=255
                B[B>0]=255
                opimg = cv2.merge([R,G,B])
                #outvid.write(opimg)
                '''imgRGB = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                lowBlue = np.array([65,105,70])
                highBlue = np.array([0,150,255])
                bmask1 = cv2.inRange(imgRGB,lowBlue,highBlue)
                lowBlue = np.array([0,0,100])
                highBlue = np.array([0,0,255])
                bmask2 = cv2.inRange(imgRGB,lowBlue,highBlue)
                bmask = bmask1+bmask2
                outputBlue = imgRGB.copy()
                outputBlue[np.where(bmask2==0)]=255
                
                #gmask = cv2.inRange(img_rgb,lowerGreen,upperGreen)
                cv2.imshow("show",outputBlue)
                outvid.write(outputBlue)'''
            if videotime > 25 and videotime < 31:
                img_hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower_red = np.array([0,50,20])
                upper_red = np.array([5,255,255])
                mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
                lower_red = np.array([170,50,20])
                upper_red = np.array([180,255,255])
                mask2 = cv2.inRange(img_hsv, lower_red, upper_red)
                mask = mask1+mask2
                output_hsv = img_hsv.copy()
                output_hsv[np.where(mask==0)] = 0
                #mask = cv2.bitwise_or(mask1, mask2 )
                #cropped = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)
                #cv2.imshow("",mask)
                outvid.write(output_hsv)
            if videotime > 30 and videotime < 36:
                img_hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower_red = np.array([0,50,20])
                upper_red = np.array([5,255,255])
                mask1 = cv2.inRange(img_hsv, lower_red, upper_red)
                lower_red = np.array([170,50,20])
                upper_red = np.array([180,255,255])
                mask2 = cv2.inRange(img_hsv, lower_red, upper_red)
                mask = mask1+mask2
                output_hsv = img_hsv.copy()
                output_hsv[np.where(mask==0)] = 0
                morphimg = output_hsv
                kernel = np.ones((15,15),np.uint8)
                #dilation = cv2.dilate(morphimg,kernel,iterations = 1)
                closing = cv2.morphologyEx(morphimg, cv2.MORPH_CLOSE, kernel)
                outvid.write(closing)
                
            
    else:
        break

print("Number of frames in the video: ", frames_counter)
video.release()
outvid.release()
cv2.destroyAllWindows() 

## Part 2 Last 24 seconds

## Last 24 seconds have been processed by manually splitting the file using the ffmpeg 
## command because of an error that I was getting when done in the above loop

## Used following ffmpeg commands to split the original video and merge the two outputs.
## ffmpeg -i CV2.mp4 -ss 00:00:35 -t 00:00:59 -c copy test.mp4
## ffmpeg -f concat -i vidlist.txt -c copy -fflags +genpts merged.mp4

filename = 'D:\\Sem 2\\CV\\test.mp4'
video = cv2.VideoCapture(filename)
outvid = cv2.VideoWriter('D:\\Sem 2\\CV\\outpytest2.mp4',cv2.VideoWriter_fourcc('D','I','V','X'), 30, (frame_width,frame_height))

colorLower = np.array([90, 50, 50])
colorUpper = np.array([150, 255, 255])

start_time = datetime.now()
while True: 
    ret, img = video.read()
    gimg = img
    hsv = cv2.cvtColor(gimg, cv2.COLOR_BGR2HSV)
    gmask = cv2.inRange(hsv,colorLower,colorUpper)
    gmask = cv2.morphologyEx(gmask, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    img[np.where(gmask==255)] = 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    ctime = datetime.now()
    elapsed = ((ctime - start_time).total_seconds())/3
    msg = "Time Elapsed : "+str(elapsed)+" seconds"
    cv2.putText(gimg,msg,(50,50), font, 1,(255,255,255),1,cv2.LINE_AA)
    outvid.write(gimg)
  
video.release()
outvid.release()
cv2.destroyAllWindows()