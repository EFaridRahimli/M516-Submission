# Classification of Gamma Rays for Cherenkov Telescope Array using ML/AI
## Farid Rahimli
## GH1029730
## M516
## Business Project in Big Data & AI
# **Project Overview**
---
#### *Notebook develops the machine-learning/AI model to classify high energy particles read by the MAGIC Gamma Telescope.*
---
#### **Business Problem**
#### *Main idea of this project is to procure highly efficient and accurate model that is able to distinguish between gamma-ray signals which are very valuable for scientific purposes named as a class (`g`) and background noise or hadronic showers that do not carry any useful information named (`h`) Enabling this classification will directly affect scientific value of the telescope as well as improve the ROI of the whole facility due to improved timeliness*
---
#### **Method**
- #### *Fetch and PreProcess the telescope data from UCI repository.*
- #### *Conduct EDA \\ Understand correlation and distribution of features.*
- #### *Construct tune and train baseline traditional model (XGBoost)*
- #### *Construct tune and train deep learning model (keral MLP)*
- #### *Compare evaluate and Conclude*
---
#### **Data Source**
The dataset used is the **MAGIC Gamma Telescope Data Set** from the UCI Machine Learning Repository.

**Source:** [https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope)](https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope)

# **P>S. Data can be fetched using ucimlrepo**
```bash
pip install ucimlrepo
```
# To import the DATASET
```python
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
magic_gamma_telescope = fetch_ucirepo(id=159) 
  
# data (as pandas dataframes) 
X = magic_gamma_telescope.data.features 
y = magic_gamma_telescope.data.targets 
  
# metadata 
print(magic_gamma_telescope.metadata) 
  
# variable information 
print(magic_gamma_telescope.variables)
```
---
### **How to use**
1 - To clone this repository:
```bash
git clone https://github.com/EFaridRahimli/M516-Submission
```
### 2 - The project was done with conda
### 3 - Install dependacies with requirements.txt (Recommende to instal tensorflow with conda -c conda-forge for better handling)
---
### Results
XGBoost yielded best reults in for this problem with accuracy of 85% and f1-score of 90% performing better than Keras deep learning MLP model.
