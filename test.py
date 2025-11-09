import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

model=tf.keras.models.load_model("cnn_model.h5")


class_names= sorted(os.listdir("train_classification"))

def predict_image(img_path):
    img=image.load_img(img_path,        target_size=(224,224))
    img_array=image.img_to_array(img)
    img_array=np.expand_dims(img_array, axis=0)
    img_array=img_array / 255.0 

   
    predictions=model.predict(img_array)
    predicted_class=np.argmax(predictions[0])
    confidence=np.max(predictions[0])*100

    print(f"Predicted Class: {class_names[predicted_class]} ({confidence:.2f}% confidence)")

if __name__ == "__main__":
    img_path = input("Enter image path: ").strip()
    predict_image(img_path)