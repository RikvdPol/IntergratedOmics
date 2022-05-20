'''
author = Hicham
'''

import pandas as pd
import sys

class Preprocessing():
    "A class to sort the dataset with the first column as target and the remaining as features"
    def __init__(self, dataset):
        self.dataset = dataset
    
    def sort_dataset(self):
        "Requests the target and features from the user and places the target as the first column"
        "and checks whether the sorted dataset is correct"
        while True:
            self.dataset = pd.DataFrame(self.dataset)
            print("Your input dataset\n")
            display(self.dataset)
            request_target = str(input("Please provide the exact target columnname.\nYour input:"))
        
            try:
                first_column = self.dataset.pop(request_target)
                self.dataset.insert(0, request_target, first_column)
                print("#####################\n\nDown below the sorted dataset is provided")
                display(self.dataset)
            except:
                print("Column name cannot be found. Please make sure to exactly copy and past the target column name.")
                continue 
                
            try:   
                check_dataset = input("#### Description ####\nThe first column of the preprocessed dataset is supposed to be your selected target column. The remaining columns should be the remaining columns as predictors.\n#####################\n\nIs this dataset correctly sorted?\nType y for Yes\nType n for No\n\nYour input:")
            except:
                print("Please only type \"y\" for Yes or \"n\" for No")
            

            if check_dataset == "y":
                print("---------------------------\nDataset is correctly preprocessed and stored\n--------------------------")
                target = self.dataset.iloc[:, 0]    # Splits the dataset into the target column
                features = self.dataset.iloc[: , 1:] # Splits the dataset into the features column(s)
                return target, features
                
            if check_dataset == "n":
                print("--------------------\nDataset is not correctly sorted.\n--------------------")
                try:
                    check_terminate = str(input("Do you want to terminate the program or start over?\nType \"t\" to terminate\nType \"s\" to start over\n"))
                except:
                    print("Please only type \"t\" to terminate the program or \"r\" to restart")
                    if check_terminate == "t":
                        quit()
                    if check_terminate == "r":
                        continue 
                        
print("Loading the data... Please wait.")    
df = pd.read_csv('Covariates.csv', index_col=0, sep="\t")   # Example file is used
target, features = Preprocessing(df).sort_dataset()   # target and features are extracted from dataset


#MetaPhlan3 
