import dlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow.compat.v1 as tf # tf 1.xx version
tf.disable_v2_behavior() # tf 2.xx version 에서도 작동하게 해주는 코드
# import tensorflow._api.v2 as tf
import numpy as np

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('./models/shape_predictor_5_face_landmarks.dat')


def align_faces(img, detector, sp):
    dets = detector(img, 1)
    objs = dlib.full_object_detections()
    for detection in dets:
        s = sp(img, detection)
        objs.append(s)
        faces = dlib.get_face_chips(img, objs, size=256, padding=0.3)
        return faces


# test_img = dlib.load_rgb_image('./imgs/02.jpg')
# test_faces = align_faces(test_img, detector, sp)
#
# fig, axes = plt.subplots(1, len(test_faces)+1, figsize=(20,16))
# axes[0].imshow(test_img)
# for i, face in enumerate(test_faces):
#     axes[i+1].imshow(face)
#
# plt.show()


sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.import_meta_graph('./models/model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./models'))
graph = tf.get_default_graph()
X = graph.get_tensor_by_name('X:0')
Y = graph.get_tensor_by_name('Y:0')
Xs = graph.get_tensor_by_name('generator/xs:0')


def preprocessing(img):
    return (img / 255. - 0.5) * 2 # 이미지 생성시 데이터 범위 (-1,1)


def deprocessing(img):
    return (img + 1) / 2 # 이미지 구분시 데이터 범위(0,1)


img1 = dlib.load_rgb_image('./imgs/12.jpg')
img1_faces = align_faces(img1, detector, sp)

img2 = dlib.load_rgb_image('./imgs/makeup/vFG56.png')
img2_faces = align_faces(img2, detector, sp)

fig, axes = plt.subplots(1, 2, figsize=(16, 10))
axes[0].imshow(img1_faces[0])
axes[1].imshow(img2_faces[0])

plt.show()

# 이미지 확인
# img = dlib.load_rgb_image('./imgs/12.jpg')
# plt.figure(figsize=(16,10))
# plt.imshow(img)
# plt.axis('off')
# plt.show()


# 얼굴 부분 네모로 표시
# img_result = img.copy()
# dets = detector(img) #얼굴의 좌표를 만들어줌 이미지 안에 얼굴이 여러개 있으면 각각의 좌표를 만듦
# if len(dets) == 0:
#     print('cannot find faces!')
# else:
#     fig, ax = plt.subplots(1,figsize=(16,10))
#     for det in dets:
#         x, y, w, h = det.left(), det.top(), det.width(), det.height()
#         rect = patches.Rectangle((x,y),w,h,linewidth=3,edgecolor='r',facecolor='none')
#         ax.add_patch(rect)
#
#     ax.imshow(img_result)
#     plt.show()


# 5_face_landmarks
# fig, ax = plt.subplots(1,figsize=(16,10))
# objs = dlib.full_object_detections()
# for detection in dets:
#     s = sp(img, detection)
#     objs.append(s)
#     for point in s.parts():
#         circle = patches.Circle((point.x, point.y), radius=3,edgecolor='r',facecolor='r')
#         ax.add_patch(circle)
#
# ax.imshow(img_result)
# plt.show()

# 얼굴만 잘라오기
# faces = dlib.get_face_chips(img, objs, size=256, padding=0.3)
# fig, axes = plt.subplots(1,len(faces)+1, figsize=(20,16))
# axes[0].imshow(img)
# for i, face in enumerate(faces):
#     axes[i+1].imshow(face)
#
# plt.show()

# 이미지에서 얼굴만 찾아서 return해주는 함수
