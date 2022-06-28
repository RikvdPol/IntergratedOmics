"""
author: Hicham Jemil
""" 

import pandas as pd
import re
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from IPython.display import display, HTML
display(HTML("<style>.container { width:85% !important; }</style>"))

class StatsCalc():
    "Class to obtain statistical parameters of the dataset and to propose a recommondation"
    "for future ML algorithms. The following details will be provided; Missing values, Outliers,"
    "skewdness, number of columns"
    def __init__(self, dataset,features):
        self.dataset = dataset
        self.features = features
        
    def return_stats(self):
        "Return stats info in a dataframe and as properties"
        describe = self.dataset.describe()
        Q1 = self.dataset.quantile(0.25) # lists the lower quantiles
        Q3 = self.dataset.quantile(0.75) # lists the higher quantiles
        IQR = Q3 - Q1  # # lists the interquatile range
        #lists all the outliers
        outliers = pd.DataFrame(((self.dataset < (Q1 - 1.5 * IQR)) | (self.dataset > (Q3 + 1.5 * IQR))).sum())
        outliers = outliers.T # Transepoe the DF to fit in the overview
        outliers_df = describe.append(outliers)
        outliers_df = outliers_df.rename(index= {0 : "OutL"})
        total_datapoints = self.dataset.count().sum()
        total_outlier = outliers_df.loc["OutL"].sum()
        total_outlier_perc = total_outlier/total_datapoints * 100
        total_missing = self.dataset.isnull().sum().sum()
        total_missing_perc = total_missing/total_datapoints * 100
        total_features = len(self.features.columns)
        
        overview_df =  pd.DataFrame({
            "Total missing values": [total_missing, total_missing_perc],
            "Total outliers": pd.Series([total_outlier, total_outlier_perc]),
            "Total datapoints": pd.Series(total_datapoints),
            "Total predictors": pd.Series(total_features)
            }).rename(index= {0 : "Absolute number", 1:"Percentage in %"})
        
        outliers_df = outliers_df.style.set_caption("Statistical overview dataset")
        return outliers_df,overview_df,total_outlier, total_missing, total_datapoints, total_features
    
class Recommondation(StatsCalc):
    def __init__(self, 
                 dataset,
                 features,
                 outliers_df, 
                 overview_df, 
                 total_outlier, 
                 total_missing,
                 total_datapoints,
                 total_features):
       # super().__init__(dataset, features)
        self.dataset = dataset
        self.features = features
        self.outliers_df = outliers_df
        self.overview_df  =  overview_df
        self.total_outlier = total_outlier
        self.total_missing  = total_missing
        self.total_datapoints = total_datapoints
        self.missing_perc = overview_df.iloc[1,0]
        self.outlier_perc = overview_df.iloc[1,1]
        self.total_features = total_features
    
    def provide_overview(self):
        "This method provides the user an overview of missing values, outliers"
        "dataset detais"
        print("Calculating statistical values of your dataset... Please wait.\n")
        print("In the table below a statistical overview of your dataset is provided")
        display(self.outliers_df)
        display(self.overview_df)
        ### Results of the recommondations will be updated in the properties down below and provided to following modules
        PCA_advice = []
        print("###### Recommondation for possible further preprocessing before feeding dataset into XGBoost:\n")
        
        # Missing values
        print("### Missing Values\nSince XGBoost can handle missing values. A very high proportion of missing values might however impact the accuracy of the model.")
        print("Your dataset has", format(self.missing_perc, ".2f"), "%", "missing values")
        print("# Recommondation")
        if self.missing_perc < 30:
            print("This is a relatively small proportion of your dataset.\nThese datapoints should be removed if there is no specific underlying reason why these datapoints are missing")
        else:
            print("This is a relatively big proportion of your dataset.\nIt is adviced to double check the perhaps underlying reason for this relatively large proportion of missing data.")
            print("If no underlying reason can be found, it is still nessary to remove the missing values since XGBoost cannot run with missing values")
        print("\n### Outliers\nXGBoost is robust to outliers. Relatively few outliers will not have a big influence on the model accuracy.\nHowever, a significant large propotion of outliers still might negatively impact the results")
        print("Therefore the proportion of outliers will be assessed prior to continuation of this program.")
        
        # Outliers
        print("Your dataset has", format(self.outlier_perc, ".2f"), "% outliers.")
        print("# Recommondation")
        if self.outlier_perc < 15:  # I have not taken into account the leve of deviation of the outlier
            print("This is a relatively small proportion of your dataset.\nThese outliers should remain in your dataset")
        else:
            print("This is a relatively large proportion of your dataset. \nIt is adviced to investigate the perhaps underlying reason of the outliers.")
            print("If no underlying reason can be found, it is adviced to continue the program with the outliers.")
            
        # Dataset size
        print("\n### Dataset size\nXGBoost is a solid machine learning algorithm. Very large datasets will however require significant run time.")
        print("It is therefore adviced to reduce the complexity of your dataset if this is the case.")
        print("You could opt for either a so-called Principal Component Anaysis algorithm\n, or a Regularization method called Elastic Net in the following steps if this is the case with your dataset.")
        print("# Recommondation")
        if self.total_features > 50:
            print("Your dataset contains relatively a large amount of predictors. A PCA preprocessing is adviced.")
            PCA_advice = ["p"]
        if self.total_features < 50:
            print("Your dataset contains relatively a small amount of predictors. A PCA or regularization is not adviced.")
            
            
        return PCA_advice
        
    def request_user_rec(self, PCA_advice):
        "Requests whether the user want to peform the recommondationns"
        "or their own specific options such as PCA, regularization or none."
        while True:
            check_recommondation = input("\n#### Would you like to proceed with recommondations of this program?\nType y for Yes\nType n for No and to specify your own options\nYour input:").lower()
            try:
                if check_recommondation == "y":
#                     PCA_advice = PCA_advice   ## I coded it like this so that if there is an error we could know where the error is derived from
                    print("Data processing will proceed according to the recommondation.")
                    return PCA_advice

                if check_recommondation == "n":
                    print("\n### Specify your selection.\nType \"p\" if you would like to perform PCA.")
                    print("Type \"r\" if you would like to perform Regulatization.")
                    print("Type \"c\" if you want to continue to XGBoost without further preprocessing.")
                    print("Note: If you want to perform both PCA and regularization, please provide your input in the desired order as follow:\n\tp,c  (if you want to perform first PCA and then regularization)")
                    while True:
                        check_user_req = input("\nYour input:").lower().strip()  #strips all the trailling and leading whitespaces and places it in a list
                        check_user_req = re.sub(r"\s", "", check_user_req)   # Use regex to remove all whitespaces 
                        if check_user_req == "p":
                            print("PCA data processing will be performd")
                            PCA_advice = [check_user_req]
                            return PCA_advice
                        if check_user_req == "r":
                            print("Regularization data processing will be performd")
                            PCA_advice = [check_user_req]
                            return PCA_advice
                        if check_user_req == "c":
                            print("Dataset will be feeded into XGboost without further data processing")
                            PCA_advice = [check_user_req]
                            return PCA_advice
                        if check_user_req == "r,p":
                            print("First regularization followed by PCA.")
                            PCA_advice = [check_user_req]
                            return PCA_advice
                        if check_user_req == "p,r":
                            print("First PCA will be performed followed by regularization.")
                            PCA_advice = [check_user_req]
                            return PCA_advice
                        else:
                            print("### Error")
                            print("Please only type for example \"p,r\" if you want to perform first PCA followed by Regularization.")
                            print("You cannot select both option c (no further processing) and r (Regularization) or p (PCA) at the same time")
                            continue
                else:
                    print("\n#### Error ####")
                    print("Please only type \"y\" for Yes and \"n\" for No.")
                    continue

            except:
                print("\n#### Error ####")
                print("Only type the indicated symbol as input")
                continue
                
def script_recommendation(df, features):
    stat_result = StatsCalc(df, features)    # Retrieve dataset and extracted features from preprocess.py
    outliers_df, overview_df, total_outlier, total_missing,total_datapoints, total_features = stat_result.return_stats()   # Retrieve stats info and display user the information
    test2 = Recommondation(df,features, outliers_df, overview_df, total_outlier, total_missing, total_datapoints, total_features)   
    advice = test2.provide_overview()  # Provide the user an overview of the dataset including the advice
    end_result = test2.request_user_rec(advice) # Propose recommendation based on input and request options from user 
    return end_result


