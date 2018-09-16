import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50, preprocess_input
from tqdm import tqdm

def save_embeddings(dataset_path, resize=(250,250)):
    img_paths = os.listdir(dataset_path)
    img_paths = [os.path.join(dataset_path, img) for img in img_paths]
    
    model = ResNet50(weights="imagenet", include_top=False, pooling="avg")
    embeddings_tensor = np.zeros((len(img_paths), 2048)) #use 1024 for MobileNet and 512 for VGG16
    
    with tqdm(total=len(img_paths)) as pbar:
        for n, img in enumerate(img_paths):
            img_array = plt.imread(img)
            img_array = cv2.resize(img_array, dsize=resize, interpolation=cv2.INTER_CUBIC)
            img_array = preprocess_input(img_array)
            img_array = model.predict(np.expand_dims(img_array, axis=0)) #default ouput shape (6,6,2048)
            embeddings_tensor[n] = img_array.flatten()
            pbar.update(1)
    embeddings_tensor = embeddings_tensor.astype(np.float16)
    
    np.save("../embeddings.npy", embeddings_tensor) #npy files save/load faster than csv when working with ndarrays

if __name__ == "__main__":
    save_embeddings("../static/images/store") #need to avoid hardcoding this path on a future version