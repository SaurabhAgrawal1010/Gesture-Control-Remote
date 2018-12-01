import cv2
import numpy as np
import copy
import math
import os
import pychromecast
from pychromecast.controllers.youtube import YouTubeController
chromecasts = pychromecast.get_chromecasts()
print(chromecasts)
cast = chromecasts[0]
print(cast.device)
cast.wait()
print(cast.status)
yt = YouTubeController()
cast.register_handler(yt)
video_id = ['Y_l9fSmG5rg', 'giYeaKsXnsI', 'QwievZ1Tx-8', 'eRapXe3Bi8o']
video_num = 0
volume_num = 0.1
cast.set_volume(volume_num)

exitAll = False
removeEscUsage = False
lCnt = 0

cam = cv2.VideoCapture(0)

cv2.namedWindow("Interactive Set Top Box")

img_counter = 0
cap_region_x_begin=0.4  # start point/total width
cap_region_y_end=0.7  # start point/total width
threshold = 60  #  BINARY threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

# variables
isBgCaptured = 0   # bool, whether the background captured
triggerSwitch = True  # if true, keyborad simulator works
prevCnt = 0
strtInteract = False
prxCor = 0
pryCor = 0
xTot = 0
yTot = 0
loopCount = 0

def removeBG(frame):
    fgmask = bgModel.apply(frame,learningRate=learningRate)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # res = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

def calculateFingers(res,drawing):  # -> finished bool, cnt: finger count
    #  convexity defect
    hull = cv2.convexHull(res, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(res, hull)
        if type(defects) != type(None):  # avoid crashing.   (BUG not found)

            cnt = 0
            for i in range(defects.shape[0]):  # calculate the angle
                s, e, f, d = defects[i][0]
                start = tuple(res[s][0])
                end = tuple(res[e][0])
                far = tuple(res[f][0])
                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # cosine theorem
                if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                    cnt += 1
                    cv2.circle(drawing, far, 8, [211, 84, 0], -1)
            return True, cnt
    return False, 0

while True:
    
    removeEscUsage = False
    
    lCnt = 0
    loopCount = 0
    xTot = 0
    yTot = 0
    prxCor = 0
    pryCor = 0
    
    while True:
        ret, frame = cam.read()
        frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
        frame = cv2.flip(frame, 1)  # flip the frame horizontally
        cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                     (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
        cv2.imshow("Interactive Set Top Box", frame)
        
        if isBgCaptured == 1:  # this part wont run until background captured
            img = removeBG(frame)
            img = img[0:int(cap_region_y_end * frame.shape[0]),
                        int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI
            #cv2.imshow('mask', img)
    
            # convert the image into binary image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #denoised_gray = cv2.fastNlMeansDenoising(gray, None, 4,7,21)
            blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
            #blur = cv2.GaussianBlur(denoised_gray, (blurValue, blurValue), 0)
            cv2.imshow('blur', blur)
            ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)
            #cv2.imshow('ori', thresh)
        
            # get the coutours
            thresh1 = copy.deepcopy(thresh)
            _,contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            length = len(contours)
            maxArea = -1
            #print('length: ',length)
            lCnt += 1
            if length == 0 and lCnt > 2 and loopCount > 1:
                removeEscUsage = True
            if length > 0:
                
                for i in range(length):  # find the biggest contour (according to area)
                    temp = contours[i]
                    area = cv2.contourArea(temp)
                    if area > maxArea:
                        maxArea = area
                        ci = i
    
                loopCount+=1
                res = contours[ci]
                hull = cv2.convexHull(res)
                if triggerSwitch is False:
                    M = cv2.moments(hull)
                    #print('M: ',M)
                    try:
                        xCor = int(M["m10"] / M["m00"])
                        yCor = int(M["m01"] / M["m00"])
                        #print(xCor)
                        #print(yCor)
                    except:
                        continue
                    if loopCount != 1:
                        xTot += (xCor - prxCor)
                        yTot += (yCor - pryCor)
                        
                        #if ((prxCor != 0) and (pryCor != 0) and (xCor == prxCor) and (yCor == pryCor)):
                        #    removeEscUsage = True
                        
                    prxCor = xCor
                    pryCor = yCor
                                        
                drawing = np.zeros(img.shape, np.uint8)
                
                cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
                cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
                
                isFinishCal,cnt = calculateFingers(res,drawing)
                #print('isFinishCal: ',isFinishCal)
                
                if triggerSwitch is True:
                    if strtInteract is False:
                        if isFinishCal is True and cnt <= 1:
                            #print (cnt)
                            if (prevCnt == 1) and (cnt == 0):
                                strtInteract = True
                            prevCnt = cnt
                
            #cv2.imshow('output', drawing)
        
        if not ret:
            break
        k = cv2.waitKey(1)
    
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            exitAll = True
            break
        elif removeEscUsage:
            print("removeEscUsage hit, closing...")
            removeEscUsage = False
            break
        elif k%256 == 98:
            # b pressed
            bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
            isBgCaptured = 1
            print( '!!!Background Captured!!!')
        elif k%256 == 32:
            # SPACE pressed
            img_name = "{}.jpg".format(img_counter)
            cv2.imwrite(img_name, thresh)
            print("{} written!".format(img_name))
            img_counter += 1
        elif k == ord('n'):
            triggerSwitch = True
            print ('!!!Trigger On!!!')
            
    print(xTot)
    print(yTot)
    
    if strtInteract is True:
        print('Start TV!!')
        #img = Image.open(directory + os.listdir(directory)[im])
        #img.show()
        #img.close
        video_num = 0
        yt.play_video(video_id[video_num])
        triggerSwitch = False
        strtInteract = False
    
    if (abs(xTot) > abs(yTot)):
        if (xTot > 0):
            print('Left to Right')
            video_num += 1
            yt.play_video(video_id[video_num])
            
        else:
            print('Right to Left')
            video_num -= 1
            yt.play_video(video_id[video_num])
            
    if (abs(yTot) > abs(xTot)):
        if (yTot > 0):
            print('Up to Down')
            volume_num -= 0.5
            cast.set_volume(volume_num)
            
        else:
            print('Down to Up')
            volume_num += 0.5
            cast.set_volume(volume_num)
    
     
    #if not ret:
     #   break
        
    #while True:
    #key = cv2.waitKey(1)
     #   if key%256 == 32:
      #      break
            
    #if key%256 == 27:
        #exitAll = True
     #   break
    
    if exitAll:
        break
        
    
cam.release()

cv2.destroyAllWindows()
cast.quit_app()
