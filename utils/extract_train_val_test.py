import numpy as np
import pandas as pd
import os
import sys
import shutil
from tqdm import tqdm

def extract_train_val_test(category, ratio, sampling=None, sampling_unmatched=None):
    customer_df = pd.read_csv("../notebooks/customer_df.csv") #modify paths accordingly
    retrieval_df = pd.read_csv("../notebooks/retrieval_df.csv")
    dataset_path = "../photos_resized" #modify paths accordingly

    avail_categories = customer_df["category"].unique()
    assert category in avail_categories, print("Please use a valid category: {}".format(" ".join(avail_categories)))

    customer_df = customer_df[customer_df["category"] == category] #filter by category
    retrieval_df = retrieval_df[retrieval_df["category"] == category]
    customer_df = customer_df.drop_duplicates(subset=["photo"]) #removing same bounding boxes to avoid bias number of matches

    merged_df = customer_df.merge(retrieval_df, how="inner", on="id", suffixes=("_cust", "_retr")) #these are how many all:all matches are available
    unmatched_df = retrieval_df[~retrieval_df['id'].isin(merged_df['id'].unique())] #dataframe for all unmatched itesm

    np.random.seed(2018) #select how many photos in each set, randomly shuffled
    #matched items
    perm = np.random.permutation(merged_df['photo_cust'].unique()) #permutes list of photos randomly
    split = int(len(merged_df['photo_cust'].unique())*(ratio[0]))
    valid, test = np.split(perm, [split])
    valid_df = merged_df[merged_df['photo_cust'].isin(valid)]
    test_df = merged_df[merged_df['photo_cust'].isin(test)]

    #unmatched items
    perm = np.random.permutation(unmatched_df['photo'])
    split = int(len(unmatched_df['photo'])*ratio[0])
    valid, test = np.split(perm, [split])
    unmatched_valid = unmatched_df[unmatched_df['photo'].isin(valid)]
    unmatched_test = unmatched_df[unmatched_df['photo'].isin(test)]

    all_df = []
    if sampling != None:
        for df in [valid_df, test_df]:
            all_df.append(df.sample(sampling, random_state=2018)) #in case we want to further reduce the sets we take a sample)
        valid_df, test_df = all_df[0], all_df[1]

    if sampling_unmatched != None:
        for df in [unmatched_valid, unmatched_test]:
            all_df.append(df.sample(sampling_unmatched, random_state=2018)) #in case we want to further reduce the sets we take a sample)
        unmatched_valid, unmatched_test = all_df[2], all_df[3]

    for step in zip([[valid_df, unmatched_valid] , [test_df, unmatched_test]], ["validation", "test"]):
        photos_list_cust = step[0][0]["photo_cust"].unique().tolist()
        photos_list_cust = [str(photo) + ".jpg" for photo in photos_list_cust]

        photos_list_retr = step[0][0]["photo_retr"].unique().tolist() + step[0][1]["photo"].tolist()
        photos_list_retr = [str(photo) + ".jpg" for photo in photos_list_retr]

        output_path = os.path.join("..", "photos_classified", category, step[1])
        if not os.path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
            os.makedirs(os.path.join(output_path, "customer"))
            os.makedirs(os.path.join(output_path, "retrieval"))

        print("Creating customer " + step[1] + " images")
        for photo in tqdm(photos_list_cust):
            shutil.copy(os.path.join(dataset_path, photo), os.path.join(output_path, "customer", photo))

        print("Creating retrieval " + step[1] + " images")
        for photo in tqdm(photos_list_retr):
            shutil.copy(os.path.join(dataset_path, photo), os.path.join(output_path, "retrieval", photo))

if __name__ == "__main__":
    extract_train_val_test(sys.argv[1], sys.argv[2], sys.argv[3]) #need to fix sampling=None parameter
