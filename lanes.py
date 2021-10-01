import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# function to make cordinates
def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def max_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def min_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2),(y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def canny(iamge):
    # Grayscaling
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    # Canny
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255,0,0), 10)
    return line_image        

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image
 
    ## FOR IMAGE
roadImages = cv2.imread('C:/Users/Acer/Desktop/finding-lanes/road_image.jpg')
lane_image = np.copy(roadImages)
canny_image = canny(lane_image)
cropped_image = region_of_interest(canny_image)
## Hough Lines
roadBoundries = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 180, np.array([]), minLineLength= 40, maxLineGap= 5)
averaged_lines = average_slope_intercept(lane_image, roadBoundries)
line_image = display_lines(lane_image, averaged_lines)
## Blending
combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
cv2.imshow('result', combo_image)
cv2.waitKey(0)

## FOR VIDEO
cap = cv2.VideoCapture("C:/Users/Acer/Desktop/finding-lanes/test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    # Hough Lines
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 180, np.array([]), minLineLength= 40, maxLineGap= 5)
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    # Blending
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('result', combo_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()

    ## FOR GIFS
roadgif = cv2.imread('C:/Users/Acer/Desktop/finding-lanes/road_image.gif')
lanegif = np.copy(roadgif)
canny_image = canny(lanegif)
cropped_image = region_of_interest(canny_image)
## Hough Lines
roadBoundries = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 180, np.array([]), minLineLength= 40, maxLineGap= 5)
averaged_lines = average_slope_intercept(lanegif, roadBoundries)
line_image = display_lines(lanegif, averaged_lines)
## Blending
combogif = cv2.addWeighted(lanegif, 0.8, line_image, 1, 1)
cv2.imshow('result', combogif)
cv2.waitKey(0)

cv2.destroyAllWindows()
