#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2 as cv
import math
from math import atan2, cos, sin, sqrt, pi
import numpy as np
from statistics import median
import imutils
import sys
import json
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

#Global Variables
colorChartData=None
 
def drawAxis(img, p_, q_, color, scale):
    p = list(p_)
    q = list(q_)
 
    ## [visualization1]
    angle = atan2(p[1] - q[1], p[0] - q[0]) # angle in radians
    hypotenuse = sqrt((p[1] - q[1]) * (p[1] - q[1]) + (p[0] - q[0]) * (p[0] - q[0]))
 
    # Here we lengthen the arrow by a factor of scale
    q[0] = p[0] - scale * hypotenuse * cos(angle)
    q[1] = p[1] - scale * hypotenuse * sin(angle)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
 
    # create the arrow hooks
    p[0] = q[0] + 9 * cos(angle + pi / 4)
    p[1] = q[1] + 9 * sin(angle + pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)
 
    p[0] = q[0] + 9 * cos(angle - pi / 4)
    p[1] = q[1] + 9 * sin(angle - pi / 4)
    cv.line(img, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), color, 3, cv.LINE_AA)

def getOrientation(pts, img):
    ## [pca]
    # Construct a buffer used by the pca analysis
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = pts[i,0,0]
        data_pts[i,1] = pts[i,0,1]
 
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
 
    # Store the center of the object
    cntr = (int(mean[0,0]), int(mean[0,1]))
    '''
    ## [visualization]
    # Draw the principal components
    cv.circle(img, cntr, 3, (255, 0, 255), 2)
    p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
    p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
    drawAxis(img, cntr, p1, (0, 255, 0), 1)
    drawAxis(img, cntr, p2, (255, 255, 0), 5)
    cv.imshow('principal components' , img)
    '''
    angle = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    return angle


# In[4]:


def checkProportion(w, h):
    ideal = 15
    actual = w/h
    print('Step 4: proportion = ', actual)
    if (abs(actual-ideal) < 15):
        return True
    else:
        return False


# In[5]:


def extractPts(x, y, w, h):
    stripLength = 7.3 #9
    pt1x = 0.4/stripLength*w + x
    pt2x = 1.5/stripLength*w + x
    pt3x = 2.6/stripLength*w + x #2.6
    pt4x = 3.7/stripLength*w + x #3.7
    pt5x = 4.8/stripLength*w + x #4.8
    pt6x = 5.9/stripLength*w + x #5.9
    pt7x = 7.1/stripLength*w + x #7.0
    pty = 0.5*h + y
    return[(pt1x, pty), (pt2x, pty), (pt3x, pty), (pt4x, pty), (pt5x, pty), (pt6x, pty), (pt7x, pty)]
    


# In[6]:


def cannyEdge(image): #image is rgb
    (H, W) = image.shape[:2]
    # convert the image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # blur the image
    blurred = cv.GaussianBlur(gray, (5, 5), 0)
    # Perform the canny operator
    canny = cv.Canny(blurred, 30, 150)

    return canny


# In[7]:


def getPixelColorRGB(image, x, y):
    (b, g, r) = image[y, x]
    return (r, g, b)


# In[8]:


def getPixelColorHSV(image, pts):
    pts_hsv = []
    imageHSV = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    for pt in pts:
        pts_hsv.append((imageHSV[pt[1], pt[0]])) #image[y, x]

    return pts_hsv


# In[9]:


def resizeImage(img, targetLengthForLongSide = 500):
    imgHt = img.shape[0]
    imgWd = img.shape[1]
    longerSide = 'imgHt'
    if (imgHt<imgWd):
        longerSide = 'imgWd'
    if (longerSide == 'imgHt'):
        ratio = targetLengthForLongSide / imgHt
        targetImgHt = targetLengthForLongSide
        targetImgWd = int(round(ratio * imgWd))
        img = cv.resize(img, (targetImgWd, targetImgHt), interpolation=cv.INTER_AREA)
    elif (longerSide == 'imgWd'):
        ratio = targetLengthForLongSide / imgWd
        targetImgHt = int(round(ratio * imgHt))
        targetImgWd = targetLengthForLongSide
        img = cv.resize(img, (targetImgWd, targetImgHt), interpolation=cv.INTER_AREA)
    return img


# In[10]:


def rgb_to_hex(rgb):
    return '%02x%02x%02x' % rgb


# In[11]:


def getChartFor(testname): 
    #testname = { 'Nitrate', 'Nitrite', 'Chlorine', 'TotalChlorine', 
    #             'GeneralHardness', 'TotalAlkalinity', 'pH'}
    global colorChartData
    if (colorChartData == None):
        print('import JSON')
        f = open('colorchart.json')
        colorChartData = json.load(f)
        f.close()
    
    strHTML = '<table style=\"width:100%\"><tr>'
    datasubset = colorChartData[testname]
    colors = []
    for key, value in datasubset.items():
        oklabel = "OK"
        if (value[4]==1):
            oklabel = "nOK"
        textcontent = str(value[0]) + "<br />" + oklabel
        rgb = (int(value[1]), int(value[2]), int(value[3]))
        bgcolorHex = rgb_to_hex(rgb)
        cellHTML = "<td bgcolor=\"#"+ bgcolorHex + "\">"+textcontent + "</td>"
        strHTML += cellHTML 
        colors.append((rgb, value[0]))
    strHTML += '</tr></table>'
    return colors, strHTML


# In[12]:


# queryRGBcolor = [r, g, b]
# refRGBcolors = [((r,g,b), label1), ((r,g,b), label2), ...]
# return ((r,g,b), labelx)
def getClosestColor(queryRGBcolor, refRGBcolors):
    minDiff = 100000000000000000
    querysRGB = sRGBColor(queryRGBcolor[0]/255, queryRGBcolor[1]/255, queryRGBcolor[2]/255)
    queryLAB = convert_color(querysRGB, LabColor)
    print ('queryLAB ', queryLAB)
    minDiffColor = None
    i = 0
    for refRGBColor in refRGBcolors:
        rgbComp = refRGBColor[0]
        refRGBCompRGB = sRGBColor(rgbComp[0]/255, rgbComp[1]/255, rgbComp[2]/255)
        refRGBCompLAB = convert_color(refRGBCompRGB, LabColor)
        label = refRGBColor[1]
        
        color_difference = delta_e_cie2000(queryLAB, refRGBCompLAB)
        print ('difference ', i, ' : ', color_difference)
        
        if (color_difference < minDiff):
            minDiff = color_difference
            minDiffColor = refRGBColor
        i += 1
    return minDiffColor


# In[13]:


# return [('Nitrate', 10), ('Nitrite', 0), ... ]
def getResults(imgName, img, pts):
    i=0
    tests = ['Nitrate', 'Nitrite', 'Chlorine', 'TotalChlorine', 
             'GeneralHardness', 'TotalAlkalinity', 'pH']
    strhtml = "<!DOCTYPE html><html><body><h1>"+ imgName + "</h1>"
    results = []
    for pt in pts:
        (r, g, b) = getPixelColorRGB(img, int(pt[0]), int(pt[1]))
        colors, html = getChartFor(tests[i])
        minDiffColor = getClosestColor([r,g,b], colors)
        strhtml += "<h1 style=\"background-color:rgb("+str(r)+","+str(g)+","+str(b)+");\">"
        strhtml += str(i)+ " --- best match: "+ str(minDiffColor[1]) + "</h1>"
        strhtml += html
        results.append((tests[i], minDiffColor[1]))
        i+=1
    strhtml += "</body></html> "
    f = open("color.html", "w") 
    f.write(strhtml)
    f.close()
    return results


# In[14]:


def findLargestContour(bw, minCtrArea=5000, maxCtrArea=70000):
    contours, _ = cv.findContours(bw, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    largestCtr = []
    largestCtrArea = 0
    for i, c in enumerate(contours):
        # Calculate the area of each contour
        area = cv.contourArea(c)

        # Ignore contours that are too small or too large
        if area < minCtrArea or area > maxCtrArea:
            continue
        if area > largestCtrArea:
            largestCtrArea = area
            largestCtr = c
           
    return largestCtr, contours


# In[15]:


def findRotateAngle(bw, img):
    listCtrAngles = []
    finalRotateAngle = 0 
    largestCtr, contours = findLargestContour(bw)
    finalRotateAngle = int(round(math.degrees(getOrientation(largestCtr, img))))
    # print ('angle ', finalRotateAngle)
    # Draw each contour only for visualisation purposes
    cv.drawContours(img, [largestCtr], 0, (0, 0, 255), 2)

    cv.imwrite('3 contourimage.jpg', img)
    print('Step 3: Written out contourimage.jpg OK')
    return finalRotateAngle, largestCtr


# In[19]:


def main(): 
    colorChartData=None
    imgName = sys.argv[1]
    #imgName = "Test Strip/hold2_1.jpg"
    
    if (len(imgName) == 0):
        print("Please provide an argument for imageName like: python ImageProcess.py myImage.jpg")
        exit(0)
        
    analyseImage(imgName)
        
def analyseImage(imgName):   
    # Step 1: Read File and resize
    img = cv.imread(imgName)
    if img is None:
        print("Error: File not found")
        exit(0)
    else:
        print("Step 1: Read Image OK")
    img = resizeImage(img, 800)
    img = cv.medianBlur(img,5)
    # cv.imshow('Input Image', img)
    results = []
    # Step 2: Create BW mask: grayscale img, then apply thresholding
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #_, bw = cv.threshold(gray, 30, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C,            cv.THRESH_BINARY,11,2)
    
    cv.imwrite('2 blackwhite.jpg', bw)
    print('Step 2: Written out blackwhite.jpg OK')

    # Step 3: Use PCA to find angle to rotate
    finalRotateAngle, largestCtr = findRotateAngle(bw, img.copy())
    print('Step 3: Final Rotate Angle = ', str(finalRotateAngle))

    # Step 4a: Select Largest Contour for processing
    if (len(largestCtr) > 0):    
        mask = np.zeros(gray.shape, dtype="uint8") # create mask for largest contour
        cv.drawContours(mask, [largestCtr], -1, 255, -1) 
        (x, y, w, h) = cv.boundingRect(largestCtr)
        imageROI = img[y:y + h, x:x + w]
        maskROI = mask[y:y + h, x:x + w]
        imageROI = cv.bitwise_and(imageROI, imageROI,mask=maskROI)
        rotated = imutils.rotate_bound(imageROI, -1*finalRotateAngle)

        #Repeat finding largest contour for rotated image
        gray2 = cv.cvtColor(rotated, cv.COLOR_BGR2GRAY)
        bw = cv.adaptiveThreshold(gray2, 255, cv.ADAPTIVE_THRESH_MEAN_C,            cv.THRESH_BINARY,11,2)
        bw2rgb = cv.cvtColor(bw, cv.COLOR_GRAY2RGB) * 255
        largestCtr, contours = findLargestContour(bw, maxCtrArea = 30000)
        cv.drawContours(bw2rgb, [largestCtr], 0, (0, 255, 255), 2)
        #cv.imshow('2nd contour', bw2rgb)
        (x, y, w, h) = cv.boundingRect(largestCtr)
        #Added for calibration
        padding = 7
        x=x-padding
        y=y+padding
        w=w-1*padding
        h=h-padding

        # Step 4b: Perform Proportion check to see if image proportion is as expected, 
        # Otherwise point extraction (hardcoded) will fail
        # Step 5: Extract Colors from points and output to html
        if checkProportion(w, h):
            print('Step 4: Check Proportion OK')
            pts = extractPts(x, y, w, h) 
            results = getResults(imgName, rotated, pts)
            print('Step 5: Output to color.html OK')
            for pt in pts:
                cv.circle(rotated, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), 2)
        else:
            print('Step 4: Check Proportion Failed')

        #Step 6: Output extracted points position for checking
        cv.drawContours(rotated, [largestCtr], 0, (0, 0, 255), 2) 
        cv.rectangle(rotated, (x,y), (x+w, y+h), (255, 0, 0), 2)
        #cv.imshow('6 final', rotated)
        cv.imwrite("6 final.jpg", rotated)
        print('Step 6: Output to final.jpg - OK')
        print('DONE')
    else:
        print('Step 4 Failed: No contour detected')
        
    #cv.waitKey(0)   
    #cv.destroyAllWindows()
    return results
    
if __name__=="__main__":
    main()


# In[ ]:




