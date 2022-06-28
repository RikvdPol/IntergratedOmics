# A software to increase machine learning interpretability

# Table of Contens
1. [Description](#Description)
2. [Data](#Data)
3. [Usability](#Usability)
4. [Installtions](#Installations)
5. [Software](#Software)
6. [Visualisations](#Visualisations)


# Description



## Data
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

<img src="Visualisations/Usage.gif" alt="pipelineGIF" height="100" width="100%">


# Software Requirements

All of the code was written in [python 3.8.13](https://www.python.org/downloads/release/python-3813/).
In addition, several modules were used in order for the code to gain proper functionality. Some packages are default 
packages in python, and these will not be mentioned here. These packages are available in the supplementary materials
in the project paper.

| Package           | Version | Usage                                                         |
|-------------------|---------|---------------------------------------------------------------|
| sklearn           | 1.0.2   | Used to implement several  machine learning algorithms.       |
| Numpy             | 1.21.5  | Uses for several calculations                                 |
| shap              | 0.39.0  | Used to calculate and visualise shapely values                |
| xgboost           | 1.5.0   | Used to perform the xgboost algorithm on the data             |            
| pandas            | 1.4.2   | Used to read data in the Preprocess module                    |
| re                | 2.2.1   | Used to drop specific columns in the Preprocess module        |
| IPYthon           | 8.3.0   | Used for it's display function                                |
| pca               | 1.8.2   | Used to perform pca and construct the scree and bi plot       |
| pathlib           | 2.3.6   | Used to write constructed plots to file given a specific path |
| composition_stats | 1.40-1  | Used for many functions in the Preprocess module              |
| matplotlib        | 3.5.1   | Used to create a variety of plots                             |
| pickle            | 4.0     | Used to read the pickled data                                 |


## Installation
Install the packages with either conda or pip.

conda:
```bash
  conda install <PACKAGE>=<VERSION>
```

pip
```bash
  pip install <PACKAGE>==<VERSION>
```


# Visualisations
The pipeline produces several plots aimed to increase the interpretability of machine learning algorithms.
Some plots will be shows here and their meanings will be shortly discussed.

![Diversity Plot](Visualisations/DiversityPlot.png)


The screeplot shows how many principal components explain which percentage of the variance in the data. In the case of the figure below, the first principal component explains approximately 50% of the variance in the data. The black line shows the cummulative explained variance per principal component. Furthermore, after seven components, nearly all the variance in the data has already been explained.

![Screeplot](Visualisations/Screeplot.png)




![Shapely Plots](Visualisations/ShapelyPlots.png)


# Future
We have created a questionnaire to help us further improve this pipeline. After using it, please don't hesitate 
to answer our questions. It will help us improve the pipeline in the future. The questionnaire can be found
[here](https://docs.google.com/forms/d/e/1FAIpQLSc_e2J3mxyiqu-RCSdUfX8M3nImsFRcippZnV-pZy27q75qNQ/viewform).

# Acknowledgements

# License
The project contains a MIT [license](LICENSE)
