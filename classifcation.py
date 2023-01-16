import tensorflow as tf
from tensorflow import keras
import cv2
import numpy as np
import time

MODEL_PATH = "rock_paper_scissors_cnn.h5"

model = tf.keras.models.load_model(MODEL_PATH)

model.summary()

class_names = ["rock", "paper", "scissors"]

#Create the data array that is inputed into the model
data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)

size = (150, 150)


# Turn on the camera
cap = cv2.VideoCapture(0)

while cap.isOpened():

   start = time.time()

   ret, img = cap.read()

   imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

   height, width, channels = img.shape

   scale_value = width / height

    # resize the image to the proper size (150 x 150)
   img_resized = cv2.resize(imgRGB, size, fx=scale_value, fy=1, interpolation=cv2.INTER_NEAREST)

   # Turn the image into a numpy array
   img_array = np.asarray(img_resized)

   # Normalize the image
   normalized_img_array = (img_array.astype(np.float32) / 127.0) - 1

   # Load the image into the array
   data[0] = normalized_img_array

   # run the inference
   prediction = model.predict(data)
   #print(prediction)

    # Take the results of the inference
   index = np.argmax(prediction)
   class_name = class_names[index]
   confidence_score = prediction[0][index]
   #print("Class: ", class_name)
   #print("Confidence score: ", confidence_score)

    # calcualation for fps counter
   end = time.time()
   totalTime = end - start

   fps = 1 / totalTime
   #print("FPS: ", fps)
   cv2.putText(img, f'FPS: {int(fps)}', (20,450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

   cv2.putText(img, class_name, (75,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
   cv2.putText(img, str(float("{:.2f}".format(confidence_score*100))) + "%", (75,100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)

   cv2.imshow('Classification Resized', img_resized)
   cv2.imshow('Classification Original', img)


   if cv2.waitKey(5) & 0xFF == 27:
      break


cv2.destroyAllWindows()
cap.release()