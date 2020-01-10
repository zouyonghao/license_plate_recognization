import cv2
from aip import AipOcr
from pytesseract import pytesseract
import time

APP_ID = '18234719'  # 刚才获取的 ID，下同
API_KEY = 'urDayfnSGLhXjeQFn53i4xde'
SECRECT_KEY = '0OfU6qtXGI5Au3El4AufGbeWqKGSFBlk'
client = AipOcr(APP_ID, API_KEY, SECRECT_KEY)


def ocr(image):
    cv2.imwrite('test.png', image)
    i = open('test.png', 'rb')
    img = i.read()
    # message = client.basicGeneral(img)
    try :
        message = client.basicAccurate(img)
    except:
        return ""

    time.sleep(0.5)

    # print(str(message))
    if (message['words_result_num'] > 0):
        return message['words_result'][0]['words']

def try_pytesseract(image):
    return pytesseract.image_to_string(image, config="-psm 6")


if __name__ == '__main__':

    from plate_localization import *

    rects = []

    image = cv2.imread("images/processed/CH2.jpg")
    rects += find_rect(image, np.array([0, 0, 0]), np.array([255, 255, 255]))

    for rect in rects:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        res_rect = get_rect_image(gray_image, rect)
        ocr(res_rect)
