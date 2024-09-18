import cv2
import numpy as np
import matplotlib.pyplot as plt

def coordinates(image, line_params):
    slope, intercept = line_params
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)

    x1 = min(max(x1, 0), image.shape[1])
    x2 = min(max(x2, 0), image.shape[1])

    return np.array([x1, y1, x2, y2])

def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope == 0:
            slope = 0.01

        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_fit_avg = np.average(left_fit, axis=0) if left_fit else None
    right_fit_avg = np.average(right_fit, axis=0) if right_fit else None
    
    left_line = coordinates(image, left_fit_avg) if left_fit_avg is not None else None
    right_line = coordinates(image, right_fit_avg) if right_fit_avg is not None else None

    return np.array([left_line, right_line]) if left_line is not None and right_line is not None else np.array([])

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lanes(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 15)
    return line_image

def region_of_interest(image):
    height = image.shape[0]
    triangle = np.array([
        [(200, height),
        (1100, height),
        (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

image = cv2.imread("lanes\\test_image.jpg")
lane_image = np.copy(image)
# cannyy = canny(lane_image)
# roi = region_of_interest(cannyy)


# lines = cv2.HoughLinesP(roi, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)

# average_lines = average_slope_intercept(lane_image, lines)

# lane_pic= display_lanes(lane_image, average_lines)

# combined = cv2.addWeighted(lane_image, 0.8, lane_pic, 1, 1)


capture = cv2.VideoCapture("lanes\\test2.mp4")
while(capture.isOpened()):
    returnval, frame = capture.read()

    if not returnval:
        print("Error: Could not read frame.")
        break

    if frame is None:
        print("Error: Empty frame.")
        continue
    cannyy = canny(frame)
    roi = region_of_interest(cannyy)

    lines = cv2.HoughLinesP(roi, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=4)
    average_lines = average_slope_intercept(frame, lines)

    lane_pic= display_lanes(frame, average_lines)
    combined = cv2.addWeighted(frame, 0.8, lane_pic, 1, 1)

    cv2.imshow("result", combined)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

capture.release()
cv2.destroyAllWindows
