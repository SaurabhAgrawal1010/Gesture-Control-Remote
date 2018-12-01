import cv2
import numpy as np
import copy
import math
import os
from tkinter import *

main = Tk()
canvas = Canvas(main, width=800, height=600)
canvas.grid(row=0, column=0)

# images
my_images = []
my_images.append(PhotoImage(file = 'img/black.gif'))
#my_images.append(PhotoImage(file = 'img/supersports.gif'))
my_images.append(PhotoImage(file = 'img/HBO.gif'))
my_images.append(PhotoImage(file = 'img/AXN.gif'))
my_images.append(PhotoImage(file = 'img/kbs.gif'))
my_images.append(PhotoImage(file = 'img/Discovery.gif'))
#self.my_images.append(PhotoImage(file = "ball3.gif"))

my_images_menu = []
my_images_menu.append(PhotoImage(file = 'img/black.gif'))
#my_images_menu.append(PhotoImage(file = 'img/supersports_menu.gif'))
my_images_menu.append(PhotoImage(file = 'img/HBO_menu.gif'))
my_images_menu.append(PhotoImage(file = 'img/AXN_menu.gif'))
my_images_menu.append(PhotoImage(file = 'img/kbs_menu.gif'))
my_images_menu.append(PhotoImage(file = 'img/Discovery_menu.gif'))

my_image_number = 0

# set first image on canvas
image_on_canvas = canvas.create_image(0, 0, anchor = NW, image = my_images[my_image_number])

# button to change image
#self.button = Button(main, text="Change", command=self.onButton)
#self.button.grid(row=1, column=0)

#----------------

#----------------------------------------------------------------------


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
    
    main.update_idletasks()
    main.update()
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
        #img.close()
        my_image_number += 1

        # return to first image
        if my_image_number == len(my_images):
            my_image_number = 0

        # change image
        canvas.itemconfig(image_on_canvas, image = my_images[my_image_number])
        main.update_idletasks()
        main.update()
        triggerSwitch = False
        strtInteract = False
    
    if (abs(xTot) > abs(yTot)):
        if (xTot > 0):
            print('Left to Right')
            my_image_number += 1

            # return to first image
            if my_image_number == len(my_images):
                my_image_number = 0

            # change image
            canvas.itemconfig(image_on_canvas, image = my_images[my_image_number])
            main.update_idletasks()
            main.update()
        else:
            print('Right to Left')
            my_image_number -= 1

            # return to first image
            if my_image_number == len(my_images):
                my_image_number = 0

            # change image
            canvas.itemconfig(image_on_canvas, image = my_images[my_image_number])
            main.update_idletasks()
            main.update()
    if (abs(yTot) > abs(xTot)):
        if (yTot > 0):
            print('Up to Down')
            canvas.itemconfig(image_on_canvas, image = my_images_menu[my_image_number])
            main.update_idletasks()
            main.update()
        else:
            print('Down to Up')
    
     
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

main.destroy()
