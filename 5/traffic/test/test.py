import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("/Users/xavier/Desktop/CS50_AI_Python/5/traffic/my_model")

image = cv2.imread("/Users/xavier/Desktop/CS50_AI_Python/5/traffic/test/speed30.ppm")
image = cv2.resize(image,(30,30))
print(image.shape)
cate = model.predict(np.array([image]))[0]

print(cate)
print("max value:", np.amax(cate))
print("most possible category:", np.where(cate == np.amax(cate))[0][0])
