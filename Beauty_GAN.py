import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#import tensorflow._api.v2 as tf
import numpy as np

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('./model/shape_predictor_5_face_landmarks.dat')

img = dlib.load_rgb_image('./imgs/12.jpg')
plt.figure(figsize=(16,10))
plt.imshow(img)
plt.axis('off')
plt.show()


img_result = img.copy()
dets = detector(img) #얼굴의 좌표를 만들어줌 이미지 안에 얼굴이 여러개 있으면 각각의 좌표를 만듦
if len(dets) == 0:
    print('cannot find faces!')
else:
    fig, ax = plt.subplots(1,figsize=(16,10))
    for det in dets:
        x, y, w, h = det.left(), det.top(), det.width(), det.hight()
        rect = patches.Rectangle((x,y),w,h,linewidth=3,edgecolor='r',facecolor='none')
        ax.add_patch(rect)

    ax.imshow(img_result)
    ax.show()

