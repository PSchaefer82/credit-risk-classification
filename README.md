# Module 20 Challenge – Supervised Machine Learning Introduction

Python script in Jupiter notebook  (.ipynb) fie
Connecting a large amount of observable information to those outcomes that we're interested in predicting through running and evaluating classification models for logistic regression. 

## Getting Started
1. Install Python on your computer
```
https://www.python.org/downloads/
```
2. Create a new environment
```
conda create -n dev python=3.10 anaconda -y
```
3. Activate the environment
```
conda activate dev
```
4. Install all the required dependencies.
```
pip install -r requirements.txt
```
5. Clone this repository to your local computer using `git clone`.
6. Open .iypnb file in Visual Studio Code.

**Covered in this assignment:**
- model and fit multiple supervised learning classification models
- create and train test datasets for supervised learning analysis
- evaluate classification algorithms using a confusion matrix and classification report
- use support vector machines (SVMs) as binary classifiers
- use decision trees and random forests as classifiers
- use KNN as a classifier
- separate data in labels and features
- create training and test data
- create prediction models
- create and edit panda databases  
- reading .csv files  

**Python Dependencies:**
- pandas  
- numpy  
- pathlib  
- sklearn.metrics
- sklearn.model_selection
- sklearn.linear_model Logistic Regression


## Purpose of Analysis
This analysis is designed to parse financial data of existing loans and determine the likelihood of defaulting.
## Financial Information used
Data includes the size of the loan, interest rate, the borrower’s income, a ratio of the borrower’s debt to their stated income, the number of accounts they hold at the bank, derogatory marks, the customer’s total debt amount, and the loan’s current status.
## Methods of Machine Learning used
The generated classification report visualizes the precision, recall, F1, and support scores for the model.

Terminology:
True Positive – TP
False Positive – FP
True Negative – TN
False Negative – FN


Scores descend from 1.0 to 0.0, 1.0 being a perfect score.



## Precision:
- scores the model’s ability to not incorrectly issue a false positive label.
- accuracy of positive predictions.
- TP / (TP + FP)
Results:
•	Class A had a prefect precision score of 1.0 
•	indicates model did not incorrectly label any false positives
•	Class B had a score of 0.84 
•	Not a great score but can be explained with the support statistics
•	Class A worked with 18765 data points while Class B had only 619
•	with a significantly smaller pool of data to work with, Class B was unable to train itself as thoroughly as Class A’s large dataset. 
## Recall:
- model’s ability to correctly identify and score all positive predictions.
- TP / (TP + FN)
Results:
•	checking to see how many true positives were identified by the models
•	Class A scored a near perfect 0.99
•	Class B scored 0.94
•	although Class B’s precision was a lower score it was still able to locate and identify 94% of true positives
## F1 Score:
- correct positive prediction percentage. 
- weighted harmonic mean combing the precision and recall values.
- a weighted F1 score is used to compare different classifier models.
- 2 * (recall * precision) / (recall + precision)
Results:
•	accuracy of true positive predictions
•	Class A with a perfect 1.0
•	Class B 0.89
•	overall accuracy of the model is an impressive 0.99.

## Support:
•	number of data points included in the classes

## Recommendation:
•	for the purposes of predicting loan defaults this model with worth recommendation
•	Class A reveals that with a large database it can operate with high expected accuracy
•	the smaller pool of data, Class B, wasn’t as successful but that is the expected outcome when compared to Class A’s larger pool of data and reinforces the recommendation for Class A



# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
* Explain what financial information the data was on, and what you needed to predict.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
* Describe the stages of the machine learning process you went through as part of this analysis.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any other algorithms).

## Results

Using bulleted lists, describe the accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1:
    * Description of Model 1 Accuracy, Precision, and Recall scores.

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:

* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.
