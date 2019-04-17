# General Machine Learning Practices Using Python

# ABSTRACT

To be added ...

<br>

# PREFACE

TO be added...

#CONTENTS
ABBREVIATIONS 7 <br>
TERMS 7<br>
1 INTRODUCTION 9 <br>
2 MACHINE LEARNING 10<br>
2.1 Supervised Learning 11<br>
2.1.1 Regression 11<br>
2.1.2 Classification 12<br>
2.2 Unsupervised Learning 12<br>
2.2.1 Clustering 13<br>
Reinforcement Learning 13<br>
3 PYTHON 14<br>
Python Modules and Packages 14<br>
3.1 Python Packages for Machine Learning 14<br>
3.1.1 Numpy 14<br>
3.1.2 Pandas 14<br>
3.1.3 Scikit Learn 15<br>
3.1.4 Matplotlib 15<br>
3.1.5 Seaborn 15<br>
4 PHASES IN MACHINE LEARNING 16<br>
4.1 Fetching Data 17<br>
4.2 Pre-processing 18<br>
4.2.1 Missing Values 18<br>
4.2.2 Normalizing Data 19<br>
Normalization / Min-Max Scaling 19<br>
Standardization 19<br>
4.2.3 Feature Generation 19<br>
4.3 Data Splitting 20<br>
4.4 Modelling 21<br>
4.5 Model Optimization/ Tuning 22<br>
5 MACHINE LEARNING IN PYTHON 24<br>

# VOCABULARY

## ABBREVIATIONS

ML = Machine Learning <br>
SVC = Support Vector Classification<br>
SVM = Support Vector Machine<br>

## TERMS

Categorical Data / Qualitative Data = Data type that are non-numerical. <br>
Classes = Prediction Values in Classification Algorithms.<br>
Continuous Data / Quantitative Data = Data type that are numerical. <br>
Data-frame = Database represented in columns and rows with labelled col-umns and indexed rows. <br>
Dependent Variable = Features column to be predicted. <br>
Domain Knowledge = Expertise in a field. For instance, ML engineer working for Nokia 5g network projects, can benefit from knowledge about internet net-working processes. <br>
Features / Variables = A column in data frame. For instance, age column, sex column, price column. <br>
Feature Engineering = Processing raw data to make it ready to be used by al-gorithms. <br>
Independent Variable = Features column that is used to predict the dependent variable.<br>
Multivariate = More than one variable. <br>
Outliers = Data in a column that is exceptional from its siblings, in ML outliers can affect results negatively. For instance,

1. An even number 2, in the list of odd numbers. (1, 2, 3, 5, 7, 9,11, 15)
2. A number 1000000 in the list of numbers less than 1000. (1, 200, 300, 999, 100, 25, 56, 465, 789, 1000000).
   Parameters = Different algorithms take several inputs that determine the result, some of which can be determined by the practitioner and some cannot.
   True Positive =
   Univariate = Single features or variable.
   <br>

# 1 INTRODUCTION

Human beings, can easily distinguish the tree from rock, can learn languages, learn to read and write, learn to drive. Everyday performance gets better and better. But how is this happening, we learn, from our surroundings, from trails and success, from failures, from companions. Every time a task is performed our brain receives feedback, learns from feedbacks and saves them for future references. <br>
Similarly, Machine Learning (ML) is a process of teaching an artificial system to learn through enormous amounts of data. It is crucial to remember the end goal of machine learning exercise is to produce a self-sufficient model for a specific task, hence the focus is to teach a model on how to learn than to perform. <br>
With the development in technologies, several programming languages have been adopted by machine learning engineers and data scientist to solve ma-chine learning problems. Some of the popular programming languages used are Python, R, Java. Python, however, is by far the most popular programming language for machine learning practices. <br>
Python is a high level, a general purpose programming language developed in 1991. Python has been used in several platforms such as desktop applica-tions, server-side scripting, machine learning, data analysis, web development. [1] <br>
In this paper, I will walk through several machine learning phases and solve the encounters using python and its popular packages and libraries. The prac-tices and codes are more conventions than rules. <br>

# 2 MACHINE LEARNING

Machine Learning is a field of computer science, that uses statistical algo-rithms and techniques to teach an artificial system an ability to learn feeding a huge amount of data. [2] <br>
Field of machine learning is huge and spread, but there are some popular use cases successfully touching the lives of billions of people.
<br>

1. Recommendation System: YouTube suggests your videos based on your personal history, likes and dislikes. Netflix users get movie recom-mendation based on user profile and histories.
2. Spam Detection: Gmail has a separate folder that says Spam, which holds all spam emails. This is a brilliant application of machine learning saving people from frauds.
3. Google Translate: It detects language automatically and translates to the language of choosing instantly.
4. Speech Recognition: Siri in Apple products, Alexa from Amazon are products of deep learning which is the field of machine learning.

<br>
Figure 1. Types of Machine Learning [5, p ]
Field of machine learning can be classified into three categories. [2]
<br>

1. Supervised Learning
2. Unsupervised Learning
3. Reinforcement Learning

## 2.1 Supervised Learning

Supervised learning in ML is the process in which prediction classes are al-ready known. In supervised learning, a learning algorithm uses generic rules to deduce specific rules [3][4]. For instance, if a plant is green and has leaf then rose is more inclined to be a plant than an animal.
Supervised Learning has been further divided into two sub-types depending upon the desired data type of prediction, Regression and Classification <br>

## 2.1.1 Regression

ML algorithms that are used to predict continuous data-type are categorized as regression learning algorithms. For instance, prediction of stock prices, real state housing prices.
Some of the popular regression algorithms are Linear Regression, Polynomial Regression, Support Vector Machine (SVM).<br>

Linear Regression
Linear Regression is the type of learning algorithms where the dependent and independent variable has linear relations.
<br>

## 2.1.2 Classification

Classification in ML is used to predict categorical data types. For instance, pre-diction of weather, directions. Some of the popular classification algorithms are Random Forest Classification, Decision Tree, Support Vector Classification (SVC), Logistic Regression. Logistic Regression, unlike the name suggests, is classification regression.
<br>

## 2.2 Unsupervised Learning

ML models where prediction class are unknown, and prediction is done based on an analysed pattern from features available. There are situations in real life where data available are unclear, in such scenario, unsupervised learning al-gorithms are used. <br>
Unsupervised learning algorithms are sub-categorized into three groups, Clus-tering, Anomaly Detection, Dimension Reduction. <br>

## 2.2.1 Clustering

As the name suggests, in clustering, predictions are clustered together in sev-eral groups based on the patterns recognized from the relationship between independent variables. Clustering algorithms are used when the prediction classes are unknown. K-Means Clustering, Agglomerative Hierarchical Clus-tering are few popular clustering algorithms.
<br>

## Reinforcement Learning

Reinforcement Learning is feedback-based learning, where algorithms receive feedbacks from the environment and adjust the parameters that affect the re-sult. [7][8] <br>
Some of the Reinforcement learning techniques are the following:
<br>

1. Markov decision process
2. Q-learning
3. Temporal Difference methods
4. Monte-Carlo methods [7]

Machine learning is a process of teaching machine how to learn by feeding a large amount of relevant data. ML can be divided into three categories, Super-vised Learning, Unsupervised Learning and Reinforcement learning.
The method is supervised learning when dependent variables are known and unsupervised learning when dependent variables are unknown, and algo-rithms responsibility to figure out prediction classes. Reinforcement learning is feedback-based learning, the algorithm is
<br>

# 3 PYTHON

Python is a high level, a general purpose programming language developed in 1991. Python has been used in several platforms such as desktop applica-tions, server-side scripting, machine learning, data analysis, web development. [1]

## Python Modules and Packages

Python modules are files in python with “.py" extensions. Packages are collec-tions of modules and packages are often referred to as namespaces. [9] <br>
Modules and Packages are prewritten set of codes aimed for reuse to reduce the workload of developers.
<br>

## 3.1 Python Packages for Machine Learning

Python has been used most widely in the field of ML. There are several pack-ages in python used for data analysis and machine learning, using all or some of them are mostly personal preference and experience. Some of them are as follows: <br>

## 3.1.1 Numpy

Numpy abbreviation for numerical python is a package for scientific computa-tion. It is used for creation and manipulation of multi-dimensional arrays, math-ematical operations like mean, medians, sum, maximum and minimum values. [10]

## 3.1.2 Pandas

In machine learning, data fetched from sources are converted into data frames and further processed by desire. For python, pandas is an open source library used for creation, manipulation of data-frames. [11]

## 3.1.3 Scikit Learn

Scikit learn is open source package for python, being used for data mining, data analysis, machine learning and data science. Scikit learns package offers various machine learning algorithms and tools like regression, classification, preprocessing, clustering and many more. [12]

## 3.1.4 Matplotlib

Matplotlib is python library for data visualization. Many statistical diagrams for instance histogram, graph, scatterplot can be drawn using matplotlib. Data Ex-ploration is an essential part of machine learning, where data visualization li-braries like Matplotlib are used. [13]

## 3.1.5 Seaborn

Seaborn is a data visualization library based on Matplotlib. Statistical plots like heatmap, violin plot, time series can be constructed easily with seaborn. [14]

# 4 PHASES IN MACHINE LEARNING

ML is a complicated process, procedures mostly depends on the data, desired output, client or employer. However, steps in achieving a near perfect model are similar.<br>
Typical ML phases can be categorized as the following. <br>

1. Fetching Data
2. Data Pre-processing
3. Data Splitting
4. Modelling
5. Model Optimization / Tuning

Sometimes, steps 1 to 3, in the list are also referred combinedly as feature en-gineering. In short, Feature engineering is a process where a raw data is pro-cessed and remodelled as per need of the project. After the feature engineer-ing data is ready to feed to an algorithm.<br>
In real world application, size of data is huge, and type also varies and is gen-erally referred as big data. Not only size but also data type can be different, da-ta can be informed of video, audio, text, image. <br>
Feature engineering is applied to cut a meaningful portion of data randomly from huge lump, make data more informative, make the whole process time and memory efficient. Hence, feature engineering is an iterative process.Domain knowledge can be a benefit when it comes to data pre-processing. <br>

<!-- TABLE 1. RAW DATA BEFORE PREPROCESSING

    Column_1	Column_2	Column_3

1       V           W            D
2       E           L            O
3       Y                        N
4       R            E           E
5                                N
TABLE 2. DATA AFTER PREPROCESSED
Column_1 Column_2 Column_3
1 V W D
2 E E O
3 R L N
4 Y L E -->

Basically, after being pre-processed data should be meaningful, free from out-liers, and ready to be fed to algorithms. In TABLE 2 is a table achieved from replacing, removing and adding values to prepare for further processing.<br>
However, some flaws of TABLE 1 and TABLE 2 are, in real world data are not this simple and small in size, after pre-processing all values being fed to algo-rithms should be converted in their numerical forms since algorithms cannot read non-numeric data.<br>

## 4.1 Fetching Data

Data are often stored in clouds or hardware in companies. As, mentioned earli-er data can be in several forms like video, audio, image, texts. In this thesis, all the data being used for experimentation are either freely available in python scikit libraries or has been created by author, all of them are in “.csv”(comma separated values). While in python, pandas can be used to read the files in several formats. <br>

## 4.2 Pre-processing

### 4.2.1 Missing Values

As mentioned in in section Feature Engineering, pre-processing is phase where raw data is processed and cooked to be fed to algorithm. Often in prac-tice, many values in data are missing hence its responsibility of ML engineer to find a solution to that issue either removing whole row or finding an optimal replacement for the missing value.
Often means, medians of the feature column are used as replacement value, if the feature is continuous. However, extreme values can affect the quality of the mean or median. <br>
For instance: A set of numbers X= [12,5,6,7,100, 90000, missing_value] <br>
So, missing_value = Mean of X = Sum of numbers/ Total numbers<br>
missing_value_1 = (12+5+6+7+100+90000)/6 =15021.67<br>
If we remove the outlier in the set which is 90000. Then,
missing_value_2 = (12+5+6+7+100)/5 = 26 <br>
The difference is huge, can affect predictive quality of model negatively.
Similar, can be the case for median. Hence, outliers should always be removed before performing operations like mean and median. <br>
Mode can be used if the feature column is a categorical. <br>
ML prediction method is also used by advanced practitioners. In this method, feature column with missing values are treated as dependent variables and using ML algorithms values are predicted to fill in the missing value.<br>

### 4.2.2 Normalizing Data

Data as raw obtained from several sources are often of various units and using them without neutralizing, can affect algorithm. Hence, it’s a common practice to normalize the continuous features. Other benefit of standardization is com-putational efficiency. Normalizing in ML process is done after splitting data into training and testing dataset. Two common methods used for normalizing data are as follows: <br>

### Normalization / Min-Max Scaling

Normalization is process of eliminating scaling numeric values from range 0 to 1, by subtracting minimum value in the list by the given value and dividing the difference by the difference between minimum and maximum value in that list or feature.
Normalization = (Given Number- Minimum Value) / (Maximum Value – Mini-mum value)
Outliers can affect the normalized values since; the calculation involves upper and lower margin in the list. Hence, outliers should be removed before normal-izing a numeric column.

### Standardization

In standardization, numbers are transformed in a manner where their mean is zero and standard deviation is 1. Therefore, numbers after application of standardization range from -1 to 1.
4.2.3 Feature Generation
Some features in a data-frames can be highly corelated and some are unnec-essary from the perspective of algorithm.
TABLE 3. SALARY OF EMPLOYEES IN COMPANY X
ID Age Salary per Year (In US \$)
1 25 150,000
2 20 60,000
3 30 175,000
4 58 180,000
5 35 120,000
<br>
In table 3, there are three columns of which ID column is unnecessary from the perspective of algorithm, its only helpful for human eyes. Hence, it should be removed before splitting data-frames. There are often multiple features in data. Merging or removing the features often requires domain knowledge and expe-rience.
<br>

## 4.3 Data Splitting

ML is the process of training algorithm by feeding data. ML algorithms analyse the correlation between features and learn from them to generalize a rule from those patterns. If we train and test our model in same data, model will fail to generalize relation which is sole purpose of ML. Since, future instances of test data in real world after deployment are unknown data is hence, divided into train and test set. More the clean processed data to train the better the model, so, It is a common practice to divide data into train set around 70 to 80 percent-age and test set around 30 to 20 percentage of original data. The key is to di-vide data randomly.

## 4.4 Modelling

After the data is split and independent columns have been normalized, data is then fed to algorithms for training. However, most important question arrives which algorithm to use?
Scikit-learn package provides a diagram that can be used as rough guide to solve the query about estimator or algorithm.
<br>

<!-- FIGURE 3. Scikit-learn algorithm cheat-sheet [15] -->

So, some key points to consider while choosing algorithm are:

1. If dependent feature is of continuous data type, use regression algo-rithms.
2. If dependent feature is of categorical data type, use regression algo-rithms.
3. If feature labels are unknown, and dependent columns cannot be iden-tified use clustering algorithms.
4. Start with very simple algorithm and climb your way up to more compli-cated algorithms. For instance, if the result is excellent using simple lin-ear or multi-linear regression algorithms do not use polynomial or Sup-port Vector Machine (SVM).
5. Use google search and forums for answers from experts.

<br>
After, the decision in type of algorithm, in python scikit learn package can be used for fetching algorithm. Algorithm comes with build in functions to train data, predict data.
<br>

## 4.5 Model Optimization/ Tuning

After the train and test of model, several reports can be gathered depending upon the type of algorithms.
In regression, most common error evaluation metrics are:
<br>

1. Mean Absolute Error (MAE): Mean of absolute value of errors
2. Mean Squared Error (MSE): Mean of squared value of errors.
3. Root Mean Square Error (RMSE): Square root of mean squared errors.
   Here, error is the difference between real value and predicted value. RMSE is widely popular once because it’s unit of measurement is same as unit of de-pendent value.
   <br>
   For classification algorithms, confusion matrix and classification report provide evaluation report.

4. Confusion Matrix: It is a table consisting of True Positives (TP), True Negative (TN), False Positive (FP) and False Negative (FN).
5. Accuracy can be calculated as:
   (TP+TN) / (Total Values)
   Talk about classification report more
   Now, optimization of model is done in various ways. Sometimes, we return to point zero i.e. pre-processing, may be remove column that seems unnecessary. Other times, internal parameters of algorithm that can be altered are altered for instance for KNN, number of neighbours is a parameter or attribute, can be al-tered to achieve better results.
   <br>
   <!-- Talk about clustering Evaluation More -->

# 5 MACHINE LEARNING IN PYTHON

# REFERENCES

[1] Wikipedia. Python Programming Language. Date of Retrieval 07.03.2019, <br>
https://en.wikipedia.org/wiki/Python_(programming_language)#Implementations<br>
[2] Wikipedia, Machine Learning Definition. Date of Retrieval 07.03.2019,<br>
https://en.wikipedia.org/wiki/Machine_learning<br>
[3] Wikipedia, Supervised Learning Definition. Date of Retrieval 07.03.2019, <br>
https://en.wikipedia.org/wiki/Supervised_learning<br>
[4] Swamynathan, M. 2017. Mastering Machine Learning with Python in Six Steps. Date of Retrieval 07.03.2019,<br>
https://learning.oreilly.com/library/view/mastering-machine-learning/9781484228661/ACoverHTML.html<br>
[5] Swamynathan, M. 2017. FIGURE 1. Mastering Machine Learning with Py-thon in Six Steps. Figure 2-9 Types of Machine Learning. Date of Retrieval 09.03.2019,<br>
https://learning.oreilly.com/library/view/mastering-machine-learning/9781484228661/A434293_1_En_2_Chapter.html
<br>
[6] Wikipedia. FIGURE 2. Date of Retrieval 09.03.2019,<br>
https://en.wikipedia.org/wiki/Linear_regression#/media/File:Linear_regression.svg<br>
[7] Swamynathan, M. 2017. Mastering Machine Learning with Python in Six Steps. Reinforcement Learning. Date of Retrieval 08.03.2019,<br>
https://learning.oreilly.com/library/view/mastering-machine-learning/9781484228661/A434293_1_En_2_Chapter.html <br>
[8] Wikipedia. Reinforcement Learning. Date of Retrieval 09.03.2019,<br>
https://en.wikipedia.org/wiki/Reinforcement_learning<br>
[9] LearnPython.Org. Python Modules and Packages. Date of Retrieval 10.03.2019,<br>
https://www.learnpython.org/en/Modules_and_Packages<br>
[10] Numpy Documentation. Numpy. Date of Retrieval 10.03.2019,<br>
http://www.numpy.org/<br>
[11] Pandas Documentation. Pandas. Date of Retrieval 10.03.2019,<br>
https://pandas.pydata.org/<br>
[12] Scikit Learn Documentation. Scikit Learn. Date of Retrieval 10.03.2019,<br>
https://scikit-learn.org/stable/index.html<br>
[13] Matplotlib Documentation. Matplotlib. Date of Retrieval 20.03.2019,<br>
https://matplotlib.org/<br>
[14] Seaborn Documentation. Seaborn. Date of Retrieval 20.03.2019<br>
https://seaborn.pydata.org/<br>
[12] Scikit Learn algorithm cheat-sheet. Figure 3. Date of Retrieval 15.04.2019<br>
https://scikit-learn.org/stable/_static/ml_map.png<br>
