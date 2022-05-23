"""
Author: Hicham Jemil

""" 

import shap
import matplotlib.pyplot as plt

class Shap():
    def __init__(self, model, X_train, feature_names):  
        "Initializes class for providing Shap values based on"
        "prior conducted XGboost algorithm"
        try: 
            self.model = model   # XGboost trained model
            self.X_train = X_train
            self.feature_names =  feature_names  # X_train.columns
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_values = self.explainer.shap_values(self.X_train)
        except Exception as a:
            print(a)
        else:
            print("XGboost results succefully stored for explainable Shap values processing")
        
        
    def Global(self):
        "Provides global results of Shapley values"
        plt_shap = shap.summary_plot(self.shap_values, #Use Shap values array
                             features=self.X_train, # Use training set features
                             feature_names=self.feature_names, #Use column names
                             show=False, #Set to false to output to folder
                             plot_size=(30,15)) 
        plt.bar = shap.plots.bar(self.shap_values)  # Bar plot
        try: 
            display(plt_shap)
            display(plt.bar)
        except Exception as b:
            print(b)
        else:
            print("Global feature contributions has been succesfully produced. This plot depicts the importance of each feature based on its predictive power in the model.")
    
                  
                 
    def Local(self):
        "Provides global results of Shapley values. Input of the local"
        "observation is requested as argument. "
        try:
            observation = str(input("Please provide the desired observation value on which the local Shapley values should be based on"))
        except Exception as c:
            print(c)
        local_plot = shap.force_plot(self.explainer.expected_value, 
                             shap_values[observation], 
                 features=self.X_train.loc[observation],
                 feature_names=self.X_train.columns,
                 show=False, matplotlib=True)
        
        try:
            display(local_plot)
        except Exception as e:
            print(e)
        else:
            print("Local feature contribution is succesfully produced. This plot depicts the importance of each feature that affects the single observation of on your input.")
                  
 
# Commence Shap values program
shap = Shap(model, X_train, feature_names)  # input data from XGboost                
display(Shap.Global())   # Provides initial global overview of Shap values

# Requests whether the user wants to display the specific feature contributions of an observation
try:
    local_decision = str(input("Would you like to obtain the feature contributions on a single observation?\nType y for Yes\nType n for No\n"))              
except:
    print("Please type only the letter \'y\' for Yes or \'n\' for no.") 
    
if local_decision == "y":
   display(Shap.Local())
if local_decision == "n":
    print("You have reached the end of this program")

                  
                  

            
            
            
   


        
        
        
        
        
        
