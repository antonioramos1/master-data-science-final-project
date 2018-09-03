import pandas as pd
import os
import sys
import shutil
from tqdm import tqdm

def extract_category(category):
    dataset_path = "./photos_resized" #modify paths accordingly
    customer_df = pd.read_csv("./customer_df.csv")
    retrieval_df = pd.read_csv("./retrieval_df.csv")
    
    available_categories = customer_df["category"].unique()
    assert category in available_categories, print("Please use a valid category: {}".format(" ".join(available_categories)))
    
    customer_category = customer_df[customer_df["category"] == category]
    retrieval_category = retrieval_df[retrieval_df["category"] == category]

    photos_list_customer = list(set(customer_category["photo"]))
    photos_list_customer = [str(photo) + ".jpg" for photo in photos_list_customer]
    
    photos_list_retrieval = list(set(retrieval_category["photo"]))
    photos_list_retrieval = [str(photo) + ".jpg" for photo in photos_list_retrieval]


    output_path = "photos_" + category
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(output_path + "/customer")
        os.makedirs(output_path + "/retrieval")
        
    print("Creating customer images")
    for photo in tqdm(photos_list_customer):
        shutil.copy(os.path.join(dataset_path, photo), os.path.join(output_path, "customer", photo))
        
    print("Creating retrieval images")
    for photo in tqdm(photos_list_retrieval):
        shutil.copy(os.path.join(dataset_path, photo), os.path.join(output_path, "retrieval", photo))

if __name__ == "__main__":
	extract_category(sys.argv[1])