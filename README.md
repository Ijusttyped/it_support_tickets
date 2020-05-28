## Lead time for IT support tickets

The goal of this project was to predict the lead time for IT support tickets.
Given a dataset with time-series data, I went through the complete data-science lifecycle using [CRISP-DM](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining).
At the end the best results were achieved with following techniques:

**Data Analysis**: 
* deep understanding of the dataset and underlying distribution was a key success factor.

**Outlier Handling**:
* a logistic regression was used to identify outliers and added the probability of being an outlier as a feature

**Imputation**: 
* K-Nearest-Neighbour algorithmn to fill missing values

**Feature Engineering:** 
*  I defined various mathematical theorems to measure the productivity, comapny affiliation and motivation of the employee working on the ticket

**Model**: a combination of 
* a logistic regression to predict the process a new ticket will go through and
* an ensemble of a BayesianRidge and a LSTM to predict the actual lead time