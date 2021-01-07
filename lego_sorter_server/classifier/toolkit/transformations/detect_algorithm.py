from lego_sorter_server.classifier.toolkit.transformations.transformation import Transformation, TransformationException
 
import cv2
import numpy as np


class DetectAlgorithm(Transformation):
    @staticmethod
    def transform(img):
        detected = detect_and_crop(img)
        if len(detected) != 1:
            raise TransformationException(F"Detected objects: {len(detected)} should be: 1",
                                          prefix=str(len(detected)))
        else:
            return detected[0]

def bbox(mask, img):
    m = np.any(mask != [0, 0, 0], axis=2)
    coords = np.argwhere(m)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1
 
    j = max(x1-x0, y1-y0)
 
    xdiff = (j-(x1-x0))
    ydiff = (j-(y1-y0))/2
    cropped = img[int((x0-xdiff)):int((x1+xdiff)), int((y0-ydiff)):int((y1+ydiff))]
    return cropped


def detect_and_crop(img):
    BLUR = 3
    CANNY_THRESH_1 = 15
    CANNY_THRESH_2 = 40
    MASK_DILATE_ITER = 6
    MASK_ERODE_ITER = 5
    MASK_COLOR = (0.0, 0.0, 1.0)  # In BGR format

    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, CANNY_THRESH_1, CANNY_THRESH_2)
    edges = cv2.dilate(edges, None)
    edges = cv2.erode(edges, None)

    contour_info = []
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c),
        ))
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    max_contour = contour_info[0]

    # Mask is black, polygon is white
    mask = np.zeros(edges.shape)
    cv2.fillConvexPoly(mask, max_contour[0], (255))

    mask = cv2.dilate(mask, None, iterations=MASK_DILATE_ITER)
    mask = cv2.erode(mask, None, iterations=MASK_ERODE_ITER)
    mask = cv2.GaussianBlur(mask, (BLUR, BLUR), 0)
    #cv2.imshow('img', mask)
    #cv2.waitKey()
    mask_stack = np.dstack([mask] * 3)  # Create 3-channel alpha mask

    cropped = bbox(mask_stack, img)
    return cropped
    #cv2.imshow('img', cropped)  # Display
    #cv2.waitKey()