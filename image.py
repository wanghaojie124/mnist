import numpy as np
import cv2
from PIL import Image


def get_pic_array(filename):
    img = Image.open(filename)
    nm = np.array(img)
    nm = nm.tolist()
    new_nm = []
    for i in nm:
        new_nm.extend(i)
    return new_nm


def save():
    test_data_file = open("mnist_test.csv", "r")
    test_data_list = test_data_file.readlines()
    test_data_file.close()

    for record in test_data_list[10:30]:
        record = [int(i) for i in record.split(',')]
        filename = str(record[0]) + ".png"
        array = np.array(record[1:])
        res = array.reshape(28, 28)
        img = Image.fromarray(res.astype(np.uint8))
        img.save(filename)


def threshold_by_otsu(img_file):
    image = cv2.imread(img_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 要二值化图像，必须先将图像转为灰度图
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print("threshold value %s" % ret)  # 打印阈值，超过阈值显示为白色，低于该阈值显示为黑色

    binary = cv2.resize(binary, (28, 28))
    binary = cv2.bitwise_not(binary)
    cv2.imshow("threshold", binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("test_8.png", binary)


if __name__ == '__main__':
    # get_pic_array("4.png")
    threshold_by_otsu("ba.png")
    # save()
    pass
