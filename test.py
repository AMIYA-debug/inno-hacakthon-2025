import os
import sys
import tensorflow as tf
import numpy as np
from PIL import Image

model=tf.keras.models.load_model("cnn_model.h5")
names=['fish','jellyfish','penguin','puffin','shark','starfish','stingray']

def predict_image(img_path):
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return
    try:
        img=Image.open(img_path).convert('RGB').resize((96,96))
        img_array=np.asarray(img).astype('float32')/255.0
        img_array=np.expand_dims(img_array,axis=0)
    except Exception as e:
        print(f"Couldn't process the image: {e}")
        return
    try:
        predictions=model.predict(img_array,verbose=0)
        predicted_class=np.argmax(predictions[0])
        confidence=np.max(predictions[0])*100
        print("\nTop predictions:")
        top_indices=np.argsort(predictions[0])[::-1][:3]
        for idx in top_indices:
            print(f"  {names[idx]}: {predictions[0][idx]*100:.2f}%")
        print(f"\nPrediction: {names[predicted_class]} ({confidence:.2f}% confidence)")
    except Exception as e:
        print(f"Prediction failed: {e}")

if __name__=="__main__":
    img_path=sys.argv[1] if len(sys.argv)>1 else input("Enter image path: ").strip()
    predict_image(img_path)
