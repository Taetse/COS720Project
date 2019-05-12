import numpy as np
import urllib
import urllib.request
import cv2 as cv


def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv.imdecode(image, cv.IMREAD_COLOR)

    return image


def detect_face(url):
    image = url_to_image(url)
    grayscale_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faceCascade = cv.CascadeClassifier(r'classifier.xml')
    faces = faceCascade.detectMultiScale(grayscale_image)
    if len(faces) > 0:
        return True
    else:
        return False


def main():
    print(detect_face(r"https://pbs.twimg.com/profile_images/639056539517624320/pKAPy0hP.jpg"))


if __name__ == '__main__':
    main()
