import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
from scipy.spatial.distance import cdist


def load_embeddings(full_path_file):
    return np.load(full_path_file)
    
class Recommender:
    def __init__(self):
        self.model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

    def recommend_user(self, user_img, embs_store, imgs_store, top_n, method="cosine"):
        img_array = plt.imread(user_img)
        img_array = cv2.resize(img_array, dsize=(250,250), interpolation=cv2.INTER_CUBIC)
        
        if len(img_array.shape) == 2: #fixes exception with B&W images having only 2 colour channels ie. 4540.jpg from street2shop
            img_array = np.array(Image.fromarray(img_array).convert("RGB"))
        if img_array.shape[2] > 3: #fixes exception with CMYK images having 4 colour channels ie. 4622.jpg from street2shop
            img_array = np.array(Image.fromarray(img_array).convert("RGB"))
            
        proc_img = preprocess_input(np.expand_dims(img_array, axis=0)) #preprocessing user image
        emb_user = self.model.predict(proc_img) #extracts vector representation (embedding)
        distance = cdist(embs_store, emb_user [0].reshape(1,-1), method) #cosine distance between user image and store images
        rank = np.argsort(distance.ravel()) #ranks by similarity, returns the indices that would sort the imgs_store list
        
        for n, i in enumerate(rank[:top_n]): #uses rank to select the similar images from the catalogue
            store_path = os.path.join(".", "images", "store", imgs_store[i])
            recommend_path = os.path.join(".", "images", "recommend", str(n)+".jpg")
            shutil.copyfile(store_path, recommend_path)
