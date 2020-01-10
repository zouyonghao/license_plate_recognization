import cv2
import numpy as np

localization_debug = False

algorithm = "canny"

def debug(str):
    if localization_debug:
        print(str)

def point_limit(point):
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0


def get_rect_image(old_image, rect):
    pic_hight, pic_width = old_image.shape[:2]
    if -1 < rect[2] < 1:
        angle = 1
    else:
        angle = rect[2]
    rect = (rect[0], (rect[1][0] + 5, rect[1][1] + 5), angle)

    box = cv2.boxPoints(rect)
    heigth_point = right_point = [0, 0]
    left_point = low_point = [pic_width, pic_hight]
    for point in box:
        if left_point[0] > point[0]:
            left_point = point
        if low_point[1] > point[1]:
            low_point = point
        if heigth_point[1] < point[1]:
            heigth_point = point
        if right_point[0] < point[0]:
            right_point = point

    if left_point[1] <= right_point[1]:  # 正角度
        new_right_point = [right_point[0], heigth_point[1]]
        pts2 = np.float32(
            [left_point, heigth_point, new_right_point])  # 字符只是高度需要改变
        pts1 = np.float32([left_point, heigth_point, right_point])
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(old_image, M, (pic_width, pic_hight))
        point_limit(new_right_point)
        point_limit(heigth_point)
        point_limit(left_point)
        return dst[int(left_point[1]):int(heigth_point[1]), int(
            left_point[0]):int(new_right_point[0])]
    elif left_point[1] > right_point[1]:  # 负角度
        new_left_point = [left_point[0], heigth_point[1]]
        pts2 = np.float32(
            [new_left_point, heigth_point, right_point])  # 字符只是高度需要改变
        pts1 = np.float32([left_point, heigth_point, right_point])
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(old_image, M, (pic_width, pic_hight))
        point_limit(right_point)
        point_limit(heigth_point)
        point_limit(new_left_point)
        return dst[int(right_point[1]):int(heigth_point[1]), int(
            new_left_point[0]):int(right_point[0])]


def find_waves(threshold, histogram):
    up_point = -1  # 上升点
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i, x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    return wave_peaks


def too_less_or_too_more_waves(original_image, rect):
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    res_rect = get_rect_image(gray_image, rect)

    # if localization_debug:
    #     cv2.imshow('res_rect', res_rect)
    #     cv2.waitKey()

    x_histogram = np.sum(res_rect, axis=1)
    if len(x_histogram) < 10:
        return False
    x_min = np.min(x_histogram)
    x_average = np.sum(x_histogram) / x_histogram.shape[0]
    x_threshold = (x_min + x_average) / 2
    wave_peaks = find_waves(x_threshold, x_histogram)
    if len(wave_peaks) < 1 or len(wave_peaks) > 5:
        debug("x fail")
        return False

    h, w = res_rect.shape
    y_histogram = np.sum(res_rect[int(h / 2) - 10:int(h / 2) + 10, :], axis=0)
    # y_histogram = np.sum(res_rect, axis=0)
    if len(y_histogram) < 10:
        debug("y fail histogram too short")
        return False
    y_min = np.min(y_histogram)
    y_average = np.sum(y_histogram) / y_histogram.shape[0]
    y_threshold = (y_min + y_average) / 2
    wave_peaks = find_waves(y_threshold, y_histogram)
    if len(wave_peaks) < 2:
        debug("y fail : " + str(wave_peaks))
        return False
    return True


def find_rect(original_image, lower_bound, upper_bound):
    min_area = 1000
    max_area = 10000

    blur = cv2.GaussianBlur(original_image, (3, 3), 0)
    hsv_image = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    # hsv_image = original_image
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    res = cv2.bitwise_and(blur, blur, mask=mask)

    if localization_debug:
        cv2.imshow('frame', blur)
        cv2.imshow('mask', mask)
        cv2.imshow('res', res)

        cv2.waitKey()

    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    # kernel = np.ones((20, 20), np.uint8)
    # img_opening = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
    # img_opening = cv2.addWeighted(res, 1, img_opening, -1, 0)
    ret, img_thresh = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if localization_debug:
        cv2.imshow('thresh', img_thresh)
        cv2.waitKey()
    # if algorithm is "canny":
    img_edge = cv2.Canny(img_thresh, 100, 200)
    # elif algorithm is "lap":
    # img_edge = cv2.Laplacian(img_thresh, cv2.CV_8UC1)
    # else:
    # img_edge = cv2.Sobel(img_thresh, cv2.CV_8UC1, 0, 1, ksize=5)

    kernel = np.ones((5, 30), np.uint8)
    img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)

    if localization_debug:
        cv2.imshow('edge', img_edge2)
        cv2.waitKey()

    contours, hierarchy = cv2.findContours(
        img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours2, hierarchy = cv2.findContours(
        img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contours += contours2

    rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if max_area < area or area < min_area:
            debug("area fail")
            continue
        rect = cv2.minAreaRect(cnt)
        if not too_less_or_too_more_waves(original_image, rect):
            continue
        # rect = (rect[0], (rect[1][0] + 15, rect[1][1] + 15), rect[2])

        area_width, area_height = rect[1]
        if area_width < area_height:
            area_width, area_height = area_height, area_width
        wh_ratio = area_width / area_height
        # debug(wh_ratio)
        if wh_ratio < 1.2 or wh_ratio > 8:
            debug("ratio fail : " + str(wh_ratio))
            continue

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
    lower_bound = np.array([11, 34, 0])
    upper_bound = np.array([34, 255, 255])
    return find_rect(image_for_yellow, lower_bound, upper_bound)


def find_green_rect(image_for_green):
    lower_bound = np.array([35, 34, 0])
    upper_bound = np.array([99, 255, 255])
    return find_rect(image_for_green, lower_bound, upper_bound)


if __name__ == '__main__':
    localization_debug = True
    # image = cv2.imread("defog_2.jpg")
    # image = cv2.imread("images/processed/test001.jpg")
    image = cv2.imread("images/processed/CH2.jpg")
    # image = cv2.imread("images/processed/fuxingjj01120051111184252515.jpg")
    # image = cv2.imread("images/updown_split/fuxingjj01120051111184536591.jpg")
    # image = cv2.imread("fog2.png")
    # image = cv2.imread("test1.png")
    rects = []

    rects.append(find_rect(image, np.array([0, 0, 0]), np.array([255, 255, 255])))
    # debug("yellow")
    # rects.append(find_yellow_rect(image))
    # debug("green")
    # rects.append(find_green_rect(image))
    # algorithm = "canny"
    # debug("blue")
    # rects.append(find_blue_rect(image))

    # algorithm = "lap"
    # debug("blue")
    # rects.append(find_blue_rect(image))
    # algorithm = "sobel"
    # debug("blue")
    # rects.append(find_blue_rect(image))
