"Author Hicham Jemil"

import pandas as pd
import sys
import os
import re
from composition_stats import multiplicative_replacement
from composition_stats import ilr
from composition_stats import ilr_inv
from composition_stats import clr
from composition_stats import clr_inv
from composition_stats import alr
from composition_stats import alr_inv
from composition_stats import closure
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

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
                print("---------------------------\nDataset is correctly sorted\n--------------------------")
                return self.dataset
                
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
                        
    def drop_column(self):
        "Requests the user whether it wants to drop a column"
        while True:
            print("\n")
            drop_answer = input("### Would you like to drop column(s)?\nType y for Yes\nType n for No\n\nYour input:")
            try:
                if drop_answer == "y":
                    print("Type the exact column(s) that you want to drop in the following way:")
                    print("Example: BMI,Age,Gender")
                    drop_col = input("Type in the columns")
                    drop_col = drop_col = re.sub(r"\s", "", drop_col).split(",")
                    end = self.dataset.drop(drop_col, axis = 1)
                    print("\n### Column(s) have been removed")
                    return pd.DataFrame(end)
                
                if drop_answer == "n":
                    return 
                
            except:
                print("\n### Invalid input. Please carefully read the input requirements.")
                continue
    
    def microbiome_modules(self):
        "Provides the user advanced microbiome preprocessing options"
        micro_answ = input("### Would you like to enable advanced microbiome transformation preprocessing options?\nType y for Yes\nType n for No\n\nYour input:").lower()
        micro_answ = re.sub(r"\s", "", micro_answ)
        if micro_answ == "y":
            print("\n### The following microbiome analysis preprocessing options are available:")
            print("1)  MR  : Replace all zeros with small non-zero values.")
            print("2)  CLR  :  Centre log ratio transformation # Requires MR #")
            print("3)  CLR-i  :  Inverse centre log ratio transformation # Requires MR #")
            print("4)  ILR  :   Isometric log ratio transformation # Requires MR #")
            print("5)  ILR-i  : Inverse isometric log ratio transformation # Requires MR #")
            print("6)  ALR  :  Additive log ratio transformation  # Requires MR # ")
            print("7)  ALR-i  :  Inverse additive log ratio transformation # Requires MR #")
            print("8)  Log10  :  Log 10 transformation # No diversity plot optional #")
            
            trans_ans = input("\n### Please provide which transformation you would like to perform as follow:\nExample first zero replacement followed by CLR: 1,2.\nYour input:")
            trans_ans = re.sub(r"\s", "", trans_ans).split(",")
            
            def log_10(df):
                return np.log(multiplicative_replacement(df))
            
            def micro_trans(choice, data):
                "Provides the user to opt for several compositional analysis transformations"
                "Several options can be choosen, which will be performed based on order provided by the user"
                data = pd.DataFrame(data)
                data_temp = data.values
                for i in choice:
                    options = {"1" : multiplicative_replacement(data_temp),
                                  "2" : clr(data_temp),
                                  "3" : clr_inv(data_temp),
                                  "4" : ilr(data_temp),
                                  "5" : ilr_inv(data_temp),
                                  "6" : alr(data_temp),
                                  "7" : alr_inv(data_temp),
                                  "8" : log_10(data_temp)}
            
                    data_temp = pd.DataFrame(options[i])
                    data_temp.columns = data.columns
                    target = data_temp.iloc[:, 0]    # Splits the dataset into the target column
                    features = data_temp.iloc[: , 1:] # Splits the dataset into the features column(s)

                return pd.DataFrame(data_temp), pd.DataFrame(target), features
            
            end_dataset, target, features = micro_trans(trans_ans, self.dataset)
            print("\n--------------------\nYour dataset has been transformed\n--------------------")
            return end_dataset, target, features

        
        if micro_answ == "n":
            print("End")
            return self.dataset, target, features    


    def diversity_plot(self, data):
        "Returns a stacked compositional plot based on the users data"
        data.iloc[:,1:] = closure(data.iloc[:,1:])
        print("Would you like to display a diversity plot?")
        request_div = input("Type y for Yes\nType n for No\nYour input:").lower()
        if request_div == "y":
            print("For which target would you like to display a diversity plot?")
            target = input("Provide the exact target(\nFor example: BMI,Age\nYour input:")
            target =  re.sub(r"\s", "", target)

            diff = data.loc[:,target].max() - data.loc[:,target].min()   # Calculate the range of values 
            diff_range = diff/10    # Devide the range of values into ten porpotions
            list_histo = [data.loc[:,target].min()]   # Start a list of all steps in the list, starting by the lowest
            for i in range(10):
                list_histo.append(diff_range + list_histo[i])

            intervals = []   #  Create list with all the bin intervals
            labels = []  # Creat a list of all the labels for the bins
            for i in range(10):
                 intervals.append([list_histo[i], list_histo[i+1]])
            for j,k in intervals:
                min_, max_ = j, k
                labels.append("{} {:.1f} - {:.1f}".format(target,min_,max_))  # Creat a label with all the ranges

            df_bins = pd.DataFrame()  # Creat a dataframe with all average bin data 
            for j,k in intervals:
                df_temp = pd.DataFrame(data.loc[(data[target] >= j) & (data[target] <= k)].mean())
                df_bins = pd.concat([df_temp, df_bins], axis=1)

            df_bins.columns = labels  #  add the lables to the bins
            df_bins = df_bins.sort_values(by= labels,ascending=False)  # Drop the first row with BMI
            df_bins.drop(index=df_bins.index[0], 
                axis=0, 
                inplace=True)

            # Plot a barplot 
            fig = df_bins.T.plot(kind='bar', 
                                 stacked=True, 
                                 legend=None, 
                                 width = 0.8,
                                 figsize = [10.4, 7.8])
            plt.ylabel("Relative abundance")
            plt.title("Diversity plot {}".format(target))
            plt.show()

            return df_bins

print("Loading the data... Please wait.")    

def script_preprocessing(df):
    "A function to run the preprocessing classes"
    sorted_data = Preprocessing(df).sort_dataset()  # target and features are extracted from dataset
    sorted_data = Preprocessing(sorted_data).drop_column()
    df_bins = []
    while True:
        try:
            end_data, target, features = Preprocessing(sorted_data).microbiome_modules()
            df_bins = Preprocessing(end_data).diversity_plot(end_data) 
            print("\n--------------------\nData preprocessing complete. Proceeding to next step.\n--------------------\n.")
            return end_data, target, features, df_bins
        except:
            print("\n--------------------\nNo diversity plot possible with the current transformations.\n--------------------")
            print("Reminder: Log-oriented transformations leading to negative numbers do not work in\ncompositional plots.")
            temp_choice = input("\nWould you like to return to the data preprocessing part to change your\ndata transformation choices to enable a diversity plot?\nType n for No\nType y for Yes\nYour input: ")
            if temp_choice == "y":
                continue
            if temp_choice == "n":
                print("\n--------------------\nProceeding without proving a diversity plot\n--------------------\n.")
                df_bins = Preprocessing(end_data).diversity_plot(end_data)
                return end_data, target, features, df_bins