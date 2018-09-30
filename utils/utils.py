import re
import cv2
from tqdm import tqdm
import os
import shutil
import numpy as np
import matplotlib.image as mpimg
from PIL import Image
from scipy.spatial.distance import cdist


def read_image(img_path):
    """ Reads images from a path into an array.
    Fixes exception with B&W images having only 2 colour channels ie. 4540.jpg.
    Fixes exception with CMYK images having 4 colour channels ie. 4622.jpg from street2shop """
    
    img_array = mpimg.imread(img_path)
    if (len(img_array.shape) == 2) or (img_array.shape[2] > 3):
            img_array = np.array(Image.fromarray(img_array).convert("RGB"))
    return img_array

def bbox_corners(img_path, df): #NEED TO CHANGE THE CUSTOMER DF; COLUMN LEFT IS ACTUALLY THE TOP COORDENATES
    """ Reads an img path, captures the photo ID (HAS TO BE UNIQUE IN THE DF) and returns the
    bounding box corners """
    img_id = re.split("(\\d+)", img_path)[-2]
    img_df = df[df["photo"] == int(img_id)] #assumes there are unique photo names in the df.photo
    top, left, width, height = (int(img_df["left"]), int(img_df["top"]), int(img_df["width"]), int(img_df["height"]))
    return top, left, width, height

def crop_image(img_path, customer_df, resize):
    top, left, width, height = bbox_corners(img_path, customer_df)
    img_array = read_image(img_path)
    image_cropped = img_array[left:left+height , top:top+width, :]    
    image_cropped = cv2.resize(image_cropped, dsize=resize, interpolation=cv2.INTER_CUBIC)
    return image_cropped

def find_paths(dataset_path):
    img_paths = os.listdir(dataset_path)
    img_paths = sorted(img_paths, key = lambda x: int(re.sub("(\\D)", "", x))) #sorts numerically rather than alphabetically
    img_paths = [os.path.join(dataset_path, img) for img in img_paths]
    return img_paths

def save_embeddings(dataset_path, file_name, model, prepocessing, shape, resize):
    imgs = find_paths(dataset_path)
    embeddings_np = np.zeros((len(imgs), shape)) #use 1024 for MobileNet and 512 for VGG16
    
    for n, img in enumerate(tqdm(imgs)):
        img_array = read_image(img)
        img_array = cv2.resize(img_array, dsize=resize, interpolation=cv2.INTER_CUBIC)
        img_array = prepocessing(img_array)
        img_array = model.predict(np.expand_dims(img_array, axis=0)) #default ouput shape (6,6,2048)
        embeddings_np[n] = img_array#.flatten() #TESTING WITHOUT FLATTEN
    embeddings_np = embeddings_np.astype(np.float16)
    np.save(file_name, embeddings_np) #npy files save/load faster than csv when working with ndarrays
    
def rank_recommendations(img_path, embs_retrieval, model, prepocessing, resize, customer_df=None, bbox=True, method="cosine"):
    if bbox == True:
        img_array = crop_image(img_path, customer_df, resize=resize) #crops to bbox and resizes
    else:
        img_array = read_image(img_path)
        img_array = cv2.resize(img_array, dsize=resize, interpolation=cv2.INTER_CUBIC)
    proc_img = prepocessing(np.expand_dims(img_array, axis=0)) #preprocess to NN format
    embs_target = model.predict(proc_img) #extracts embedding

    distance = cdist(embs_retrieval, embs_target.reshape(1,-1), method) #distances from user photo to retrieval
    rank = np.argsort(distance.ravel()) #ranks by similarity, returns the indices that would sort the distance array
    return rank
    
def load_embeddings(full_path_file):
    return np.load(full_path_file)
    
def recommend_user(store_database, rank, top_n):        
    ranked_database = store_database.iloc[rank] #orders database by similarity position
    ranked_database = ranked_database.drop_duplicates(subset=["id"], keep="first") #removes duplicate ids keeping only the most similar photo for each product
    ranked_list = ranked_database["photo"].values.tolist()[:top_n]
    
    for n, photo_name in enumerate(ranked_list): #uses rank to select the similar images from the catalogue
        store_path = os.path.join(".", "static", "images", "store", str(photo_name)+".jpg")
        recommend_path = os.path.join(".", "static", "images", "recommend", str(n)+".jpg")
        shutil.copyfile(store_path, recommend_path)