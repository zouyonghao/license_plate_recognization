import os

from plate_localization import *
from ocr import *

original_dir = "images/processed"
output_dir = "images/boxed"

with os.scandir(original_dir) as entries:
    for entry in entries:
        image = cv2.imread(original_dir + "/" + entry.name)
        rects = []

        # algorithm = "canny"
        # print("normal")
        rects += find_rect(image, np.array([0, 0, 0]), np.array([255, 255, 255]))

        # if len(rects) < 1:
        # print("yellow")
        rects += find_yellow_rect(image)
        # print("green")
        rects += find_green_rect(image)
        # print("blue")
        rects += find_blue_rect(image)
        #
        # algorithm = "Laplacian"
        # print("white")
        # rects += find_rect(image, np.array([0, 0, 0]), np.array([255, 255, 255]))
        # print("yellow")
        # rects += find_yellow_rect(image)
        # print("green")
        # rects += find_green_rect(image)
        # print("blue")
        # rects += find_blue_rect(image)

        print("image :" + entry.name)
        for rect in rects:

            # gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # res_rect = get_rect_image(gray_image, rect)
            # message = try_pytesseract(res_rect)

            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            res_rect = get_rect_image(gray_image, rect)
            message = ocr(res_rect)

            if message is None or len(message) < 3:
                # print("no message")
                continue
            print("get number:" + str(message))

            box = cv2.boxPoints(rect)
            box = np.int0(box)
            image = cv2.drawContours(image, [box], 0, (0, 0, 255), 1)
            # cv2.imshow("box", image)
            # cv2.waitKey()
        cv2.imwrite(output_dir + "/" + entry.name, image)
