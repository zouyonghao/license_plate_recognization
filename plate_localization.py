import cv2
import numpy as np

localization_debug = False


def find_rect(original_image, lower_bound, upper_bound):
    min_area = 2000
    max_area = 10000
    hsv_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2HSV)
    # hsv_image = original_image
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    res = cv2.bitwise_and(original_image, original_image, mask=mask)

    if localization_debug:
        cv2.imshow('frame', original_image)
        cv2.imshow('mask', mask)
        cv2.imshow('res', res)

        cv2.waitKey()

    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((20, 20), np.uint8)
    img_opening = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
    img_opening = cv2.addWeighted(res, 1, img_opening, -1, 0)
    ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img_edge = cv2.Canny(img_thresh, 100, 200)
    kernel = np.ones((10, 40), np.uint8)
    img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)

    # cv2.imshow('edge', img_edge2)
    # cv2.waitKey()

    contours, hierarchy = cv2.findContours(
        img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if max_area < area or area < min_area:
            continue
        rect = cv2.minAreaRect(cnt)
        # rect = (rect[0], (rect[1][0] + 15, rect[1][1] + 15), rect[2])
        rects.append(rect)

        if localization_debug:
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            box_img = cv2.drawContours(original_image, [box], 0, (0, 0, 255), 1)
            cv2.imshow("box", box_img)
            cv2.imwrite("box_img.png", box_img)
            cv2.waitKey()
    return rects


def find_blue_rect(image_for_blug):
    lower_bound = np.array([99, 34, 0])
    upper_bound = np.array([140, 255, 255])
    return find_rect(image_for_blug, lower_bound, upper_bound)


def find_yellow_rect(image_for_yellow):
    lower_bound = np.array([11, 34, 50])
    upper_bound = np.array([34, 255, 85])
    return find_rect(image_for_yellow, lower_bound, upper_bound)


def find_green_rect(image_for_green):
    lower_bound = np.array([35, 34, 0])
    upper_bound = np.array([99, 255, 255])
    return find_rect(image_for_green, lower_bound, upper_bound)


if __name__ == '__main__':
    localization_debug = True
    # image = cv2.imread("defog_2.jpg")
    image = cv2.imread("defog_3.jpg")
    # image = cv2.imread("fog2.png")
    # image = cv2.imread("test1.png")

    find_yellow_rect(image)
    find_blue_rect(image)
    find_green_rect(image)
