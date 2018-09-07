import pandas as pd
import os
import sys
import shutil
from tqdm import tqdm

def extract_train_val_test(category, ratio, sampling=None):
    customer_df = pd.read_csv("./customer_df.csv") #modify paths accordingly
    retrieval_df = pd.read_csv("./retrieval_df.csv")

    avail_categories = customer_df["category"].unique()
    assert category in avail_categories, print("Please use a valid category: {}".format(" ".join(avail_categories)))

    
    customer_df = customer_df[customer_df["category"] == category] #filter by category
    retrieval_df = retrieval_df[retrieval_df["category"] == category]

    customer_df = customer_df.drop_duplicates(subset=["photo"]) #removing same bounding boxes to avoid bias number of matches

    merged_df = customer_df.merge(retrieval_df, how="inner", on="id", suffixes=("_cust", "_retr")) #these are how many all:all matches are available
    
    n_ratio = int(len(merged_df)*ratio[0]) #how many photos in each test, rounding down to keep same number for both sets
    
    valid_df = merged_df.sample(n_ratio, random_state=2018) #randomly splits sets
    dif_index = list((merged_df.index).difference(valid_df.index)) #difference in indices to take the remaining photos
    test_df = merged_df.loc[dif_index[:n_ratio]] #indexing again on dif_index to ensure same number of photos in both sets
    
    if sampling != None:
        valid_df = valid_df.sample(sampling, random_state=2018) #in case we want to further reduce the sets we take a sample
        test_df = test_df.sample(sampling, random_state=2018)    

    return valid_df, test_df

def move_files(df, set_name, category):
    dataset_path = "./photos_resized" #modify paths accordingly
    photos_list_cust = df["photo_cust"].unique().tolist()
    photos_list_cust = [str(photo) + ".jpg" for photo in photos_list_cust]
    
    photos_list_retr = df["photo_retr"].unique().tolist()
    photos_list_retr = [str(photo) + ".jpg" for photo in photos_list_retr]

    output_path = os.path.join(".", "photos_classified", category, set_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, "customer"))
        os.makedirs(os.path.join(output_path, "retrieval"))
        
    print("Creating customer " + set_name + " images")
    for photo in tqdm(photos_list_cust):
        shutil.copy(os.path.join(dataset_path, photo), os.path.join(output_path, "customer", photo))
        
    print("Creating retrieval " + set_name + " images")
    for photo in tqdm(photos_list_retr):
        shutil.copy(os.path.join(dataset_path, photo), os.path.join(output_path, "retrieval", photo))

if __name__ == "__main__":
    valid_df, test_df = extract_train_val_test(sys.argv[1], sys.argv[2], sys.argv[3])
    move_files(valid_df, "validation", sys.argv[1])
    move_files(test_df, "test", sys.argv[1])