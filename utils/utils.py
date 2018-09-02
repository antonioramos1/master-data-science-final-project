import os
import sys
import shutil
import pandas as pd
import numpy as np
from keras.applications.inception_v3 import InceptionV3 , preprocess_input
from keras.models import Model
from keras.preprocessing import image
from scipy.spatial.distance import cdist


def load_embeddings(full_path_file):
    embs_loaded = np.loadtxt(open(full_path_file,'rb'),delimiter=",",skiprows=0)
    return embs_loaded

class EmbeddingsAll:
    def __init__(self):
        base_model = InceptionV3(weights="imagenet", include_top=False)
        self.model = Model(inputs=base_model.input, outputs=base_model.output)
        # Bug fix for async not working, see: https://github.com/keras-team/keras/issues/2397
        self.model._make_predict_function()

    def preprocess_img(catalogue_path):
        img_paths = os.listdir(catalogue_path)
        raw_imgs = [image.load_img(os.path.join(catalogue_path, img_path), target_size=(250,250)) for img_path in img_paths]
        proc_imgs = np.array([preprocess_input(np.expand_dims(image.img_to_array(img), axis=0)[0]) for img in raw_imgs])
        return proc_imgs

    def embeddings(self, img_arrays, output_path=None, write=False):
        if write==True:
            embeddings_img = np.array([model.predict(np.expand_dims(img, axis=0)).flatten() for img in img_arrays])
            print("writing to file")
            pd.DataFrame(embeddings_img,columns=None).to_csv(
                    os.path.join(output_path, "embeddings.csv"), index=False, header=False)
        else:
            embs_catalogue = np.array([self.model.predict(np.expand_dims(img, axis=0)).flatten() for img in img_arrays])
            return embs_catalogue

    def top_similar(self, target_img, recomd_path, embs_catalogue, imgs_catalogue, top_n, method="cosine"):
        target_img = "/".join(target_img.split("/")[:3])
        proc_imgs = EmbeddingsAll.preprocess_img(target_img) #preprocessing target folder
        embs_target = EmbeddingsAll.embeddings(self, proc_imgs) #embedding
        distance = cdist(embs_catalogue, embs_target[0].reshape(1,-1), method) #cosine distance between target and catalogue
        rank = np.argsort(distance.ravel()) #ranks by similarity, returns the indices that would sort the distance array

        for n, i in enumerate(rank[:top_n]): #uses rank to select the similar images from the catalogue
            shutil.copyfile("./images/catalogue/"+imgs_catalogue[i], recomd_path+str(n)+".jpg")
