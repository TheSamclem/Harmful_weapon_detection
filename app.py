# import the opencv library
import cv2
import numpy as np
import tensorflow as tf
import winsound



from pygame import mixer#if this module in not installed then install it by pip install pygame
mixer.init()
mixer.music.load('sound.mp3')


global label_names

model = tf.keras.models.load_model("models\\deeplearning_model.h5") 
class_list=sorted(['Dangerous','no weapon'])
label_names = sorted(class_list)
  
# define a video capture object
vid = cv2.VideoCapture(0)


def wins(ct):
    freq = 100
    dur = 50
    
    # loop iterates 5 times i.e, 5 beeps will be produced.
    for i in range(0, ct):   
        winsound.Beep(freq, dur)   
        freq+= 100
        dur+= 50
def predict(image, returnimage = False,  scale = 0.9):
   
  processed_image = preprocess(image)
  results = model.predict(processed_image)
 
  label, (x1, y1, x2, y2), confidence = postprocess(image, results)
  cv2.rectangle(image, (x1,y1), (x2,y2), (0, 255, 100), 2)
  cv2.putText(
      image, 
      '{}'.format(label, confidence), 
      (x1, y2+(50)), 
      cv2.FONT_HERSHEY_COMPLEX, scale,
      (200, 300, 100),2)
    

  print(label)
  if label=='Dangerous':
        wins(2)
        mixer.music.play()
  

        

def preprocess(img, image_size = 300):
   
    image = cv2.resize(img, (image_size, image_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype("float") / 255.0
    # Expand dimensions as predict expect image in batches
    image = np.expand_dims(image, axis=0) 
    return image

def postprocess(image, results):
 
    # Split the results into class probabilities and box coordinates
    bounding_box, class_probs = results
    class_index = np.argmax(class_probs)
   
    # Use this index to get the class name.
    class_label = label_names[class_index]

    h, w = image.shape[:2]
 
    # Extract the Coordinates
    x1, y1, x2, y2 = bounding_box[0]
 
    # Convert the coordinates from relative (i.e. 0-1) to actual values
    x1 = int(w * x1)
    x2 = int(w * x2)
    y1 = int(h * y1)
    y2 = int(h * y2)
 
    # return the lable and coordinates
    return class_label, (x1,y1,x2,y2),class_probs
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
  
    # Display the resulting frame
    cv2.imshow('frame', frame)
    # print(frame)

    # image = cv2.imread(frame)
    predict(frame, scale = 1)  




    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()