import os

import cv2

original_dir = "images/updown"
output_dir = "images/updown_split"

with os.scandir(original_dir) as entries:
    for entry in entries:
        updown_image = cv2.imread(original_dir + "/" + entry.name)
        h, w, _ = updown_image.shape
        updown_image = updown_image[int(h / 2):h, :]
        # cv2.imshow("", updown_image)
        # cv2.waitKey()
        cv2.imwrite(output_dir + "/" + entry.name, updown_image)
