# A software to increase machine learning interpretability

# Table of Contens
1. [Description](#Description)
2. [Data](#Data)
3. [Usability](#Usability)
4. [Software](#Software)
5. [Visualisations](#Visualisations)


# Description



# Data
Data has been collected and anonymized by the researchers at the UMCG. 

Data used in this project comes from already published studies and therefore data has been collected and has been made publicly available. Datasets used in this project are the following: 

-  Microbiome data: Proportions of bacteria abundances and microbial pathways from 1179 stools samples from the LifeLines-Deep cohort at the European Genome-phenome Archive under accession EGAS00001001704. 

- Metabolomics: blood metabolites measured using targeted metabolomics.  

- Proteomics: https://www.olink.com  

The gut microbiome data is collected by deep sequencing the gut micro biomes of 1135 Dutch participants, gathered from 1179 stool samples from lifelines-DEEP participants. The gut microbiome was analysed using a HiSeq2000, conducting a paired-end metagenomic shotgun sequencing (MGS). 
Using MetaPhlAn 2.0 , the microbiome sequence reads were mapped to ~1 million microbial-taxonomy- specific marker genes. The main goal of the data is to find correlations between dietary intake and diseases. In addition, metabolomics, proteomics and some anthropometric data has been gathered.
More information about the data collection can be found [here](https://ega-archive.org/studies/EGAS00001005027). 
The LifeLines-DEEP meta genomics sequencing data are freely available online. 

# Usability

# Software

All of the code was written in [python 3.8.13](https://www.python.org/downloads/release/python-3813/).
In addition, several modules were used in order for the code to gain proper functionality. Some packages are default 
packages in python, and these will not be mentioned here. These packages are available in the supplementary materials
in the project paper.

| Package | Version | Usage                                                    |
|---------|---------|----------------------------------------------------------|
| sklearn | 1.0.2   | Used to implement several  machine learning algorithms.  |
| Numpy   | 1.21.5  | Uses for several calculations                            |
| Logging | 3.8.13  | Used to create logfile explaining events during code run |
| shap    |         |                                                          |
| xgboost |         |                                                          |


# Visualisations


![Diversity Plot](Visualisations/DiversityPlot.png)

![Screeplot](Visualisations/Screeplot.png)

![Shapely Plots](Visualisations/ShapelyPlots.png)


# Future
We have created a questionnaire to help us further improve this pipeline. After using it, please don't hesitate 
to answer our questions. It will help us improve the pipeline in the future. The questionnaire can be found
[here](https://docs.google.com/forms/d/e/1FAIpQLSc_e2J3mxyiqu-RCSdUfX8M3nImsFRcippZnV-pZy27q75qNQ/viewform).