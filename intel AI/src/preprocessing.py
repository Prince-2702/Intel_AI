import cv2

def preprocess(image):
    image = cv2.resize(image,(224,224))
    image = image /255.0
    return image 