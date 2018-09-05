import pandas as pd
import os
import sys
import shutil
from tqdm import tqdm

def extract_train_val_test(category):
    dataset_path = "../photos_resized" #modify paths accordingly
    customer_df = pd.read_csv("../notebooks/customer_df.csv")
    retrieval_df = pd.read_csv("../notebooks/retrieval_df.csv")

    available_categories = customer_df["category"].unique()
    assert category in available_categories, print("Please use a valid category: {}".format(" ".join(available_categories)))

    customer_category = customer_df[customer_df["category"] == category]
    retrieval_category = retrieval_df[retrieval_df["category"] == category]

    split = int(len(customer_category['id'].unique())*0.5)
    ids_val = customer_category['id'].unique()[:split].tolist()
    ids_test = customer_category['id'].unique()[split:].tolist()

    ids_other = []
    for i in retrieval_category['id']:
        if i not in ids_val and i not in ids_test:
            ids_other.append(i)

    ids_other = list(set(ids_other))
    split_others = int(len(ids_other)*0.5)

    ids_val = ids_val + ids_other[:split_others]
    ids_test = ids_test + ids_other[split_others:]

    customer_category_val = customer_category[customer_category['id'].isin(ids_val)]
    customer_category_test = customer_category[customer_category['id'].isin(ids_test)]

    retrieval_category_val = retrieval_category[retrieval_category['id'].isin(ids_val)]
    retrieval_category_test = retrieval_category[retrieval_category['id'].isin(ids_test)]


    photos_list_customer_val = list(set(customer_category_val["photo"]))
    photos_list_customer_val = [str(photo) + ".jpg" for photo in photos_list_customer_val]

    photos_list_customer_test = list(set(customer_category_test["photo"]))
    photos_list_customer_test = [str(photo) + ".jpg" for photo in photos_list_customer_test]

    photos_list_retrieval_val = list(set(retrieval_category_val["photo"]))
    photos_list_retrieval_val = [str(photo) + ".jpg" for photo in photos_list_retrieval_val]

    photos_list_retrieval_test = list(set(retrieval_category_test["photo"]))
    photos_list_retrieval_test = [str(photo) + ".jpg" for photo in photos_list_retrieval_test]


    output_vald = "../photos_classified/" + category + "/validation/"
    if not os.path.exists(output_vald):
        os.makedirs(output_vald, exist_ok=True)
        os.makedirs(output_vald + "/customer")
        os.makedirs(output_vald + "/retrieval")

    output_test = "../photos_classified/" + category + "/test/"
    if not os.path.exists(output_test):
        os.makedirs(output_test, exist_ok=True)
        os.makedirs(output_test + "/customer")
        os.makedirs(output_test + "/retrieval")

    print("Creating customer validation images")
    for photo in tqdm(photos_list_customer_val):
        shutil.copy(os.path.join(dataset_path, photo), os.path.join(output_vald, "customer", photo))

    print("Creating customer test images")
    for photo in tqdm(photos_list_customer_test):
        shutil.copy(os.path.join(dataset_path, photo), os.path.join(output_test, "customer", photo))

    print("Creating retrieval validation images")
    for photo in tqdm(photos_list_retrieval_val):
        shutil.copy(os.path.join(dataset_path, photo), os.path.join(output_vald, "retrieval", photo))

    print("Creating retrieval test images")
    for photo in tqdm(photos_list_retrieval_test):
        shutil.copy(os.path.join(dataset_path, photo), os.path.join(output_test, "retrieval", photo))

if __name__ == "__main__":
	extract_train_val_test(sys.argv[1])
