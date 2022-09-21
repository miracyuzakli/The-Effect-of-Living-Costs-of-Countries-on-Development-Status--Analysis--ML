# -*- coding: utf-8 -*-
"""
@author: mirac
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


Data = pd.read_csv("data.csv")


## Data Cleaning and Editing
Data = Data[Data != "?"]
Data.dropna(inplace = True)
Data.reset_index(inplace = True, drop = True)

for i in Data: # Data columns
    for j in range(Data.index.size): # Data index
        Data[i].values[j] = Data[i].values[j].replace("$", "").replace(",", "")
        
        
        
for i in Data:
    if i != "CountryName":
        Data = Data.astype({i: float})
        
Development = pd.read_csv("DevelopmentData.csv") # Undeveloped = 0, Underdeveloped = 1, Developed = 2
        
Data["Development"] = Development["Development"]
        
Data.dropna(inplace = True)
Data.reset_index(inplace = True, drop = True)       
        


numData = Data.copy()

numData = numData.select_dtypes(include = ["float64", "int64"])


numData = Data.copy()

numData = numData.select_dtypes(include = ["float64", "int64"])
for i in numData:
    column = numData[i]
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)

    IRQ = Q3 - Q1
    lowerBound = Q1 - 1.5*IRQ
    upperBound = Q3 + 1.5*IRQ
    
    
    Data = Data[~Data.index.isin(((column < (lowerBound)) | (column > (upperBound))))]
Data.reset_index(inplace = True, drop = True)



## Data Visualization and Analysis
DevelopmentMean = Data.groupby("Development").mean()
DevelopmentMedian = Data.groupby("Development").median()

plt.style.use('default')


### Restaurants
column = ["Meal, Inexpensive Restaurant",
            "Meal for 2 People, Mid-range Restaurant, Three-course",
            "McMeal at McDonalds (or Equivalent Combo Meal)",
            "Cappuccino (regular)",
            "Coke/Pepsi (0.33 liter bottle)",
            "Water (0.33 liter bottle)"]
dataMean = DevelopmentMean[column]
dataMedian = DevelopmentMedian[column]
arangeSize = len(column)
plt.figure(figsize = (20, 10))

plt.subplot(1, 2, 1)
plt.bar(np.arange(arangeSize) + 0.00, dataMean.iloc[0].values, color = "#004c6d", width = 0.25)
plt.bar(np.arange(arangeSize) + 0.25, dataMean.iloc[1].values, color = "#5385a8", width = 0.25)
plt.bar(np.arange(arangeSize) + 0.50, dataMean.iloc[2].values, color = "#95c4e8", width = 0.25)
plt.xticks([r + 0.25 for r in range(len(column))], column, fontsize = 15)
plt.xticks(rotation = 90)
plt.legend(["Undeveloped", "Underdeveloped", "Developed"], fontsize = 14)
plt.ylabel("Value", fontsize = 14)
plt.title("Mean of the restaurant values", fontsize = 14)

plt.subplot(1, 2, 2)
plt.bar(np.arange(arangeSize) + 0.00, dataMedian.iloc[0].values, color = "#488f31", width = 0.25)
plt.bar(np.arange(arangeSize) + 0.25, dataMedian.iloc[1].values, color = "#75a760", width = 0.25)
plt.bar(np.arange(arangeSize) + 0.50, dataMedian.iloc[2].values, color = "#9fc08f", width = 0.25)
plt.xticks([r + 0.25 for r in range(len(column))], column, fontsize = 15)
plt.xticks(rotation = 90)
plt.legend(["Undeveloped", "Underdeveloped", "Developed"], fontsize = 14)
plt.ylabel("Value", fontsize = 14)
plt.title("Median of the restaurant values", fontsize = 14)



### Markets
column = ["Milk (regular), (1 liter)",
            "Loaf of Fresh White Bread (500g)",
            "Rice (white), (1kg)",
            "Eggs (regular) (12)",
            "Local Cheese (1kg)",
            "Chicken Fillets (1kg)",
            "Apples (1kg)",
            "Banana (1kg)",
            "Oranges (1kg)",
            "Tomato (1kg)",
            "Potato (1kg)",
            "Onion (1kg)",
            "Lettuce (1 head)",
            "Water (1.5 liter bottle)",
            "Cigarettes 20 Pack (Marlboro)"]
dataMean = DevelopmentMean[column]
dataMedian = DevelopmentMedian[column]
arangeSize = len(column)
plt.figure(figsize = (30, 10))

plt.subplot(1, 2, 1)
plt.bar(np.arange(arangeSize) + 0.00, dataMean.iloc[0].values, color = "#004c6d", width = 0.25)
plt.bar(np.arange(arangeSize) + 0.25, dataMean.iloc[1].values, color = "#5385a8", width = 0.25)
plt.bar(np.arange(arangeSize) + 0.50, dataMean.iloc[2].values, color = "#95c4e8", width = 0.25)
plt.xticks([r + 0.25 for r in range(len(column))], column, fontsize = 15)
plt.xticks(rotation = 90)
plt.legend(["Undeveloped", "Underdeveloped", "Developed"], fontsize = 14)
plt.ylabel("Value", fontsize = 14)
plt.title("Mean of the market values", fontsize = 14)

plt.subplot(1, 2, 2)
plt.bar(np.arange(arangeSize) + 0.00, dataMedian.iloc[0].values, color = "#488f31", width = 0.25)
plt.bar(np.arange(arangeSize) + 0.25, dataMedian.iloc[1].values, color = "#75a760", width = 0.25)
plt.bar(np.arange(arangeSize) + 0.50, dataMedian.iloc[2].values, color = "#9fc08f", width = 0.25)
plt.xticks([r + 0.25 for r in range(len(column))], column, fontsize = 15)
plt.xticks(rotation = 90)
plt.legend(["Undeveloped", "Underdeveloped", "Developed"], fontsize = 14)
plt.ylabel("Value", fontsize = 14)
plt.title("Median of the market values", fontsize = 14)


### Transportation
column = ["One-way Ticket (Local Transport)",
            "Monthly Pass (Regular Price)",
            "Taxi Start (Normal Tariff)",
            "Taxi 1km (Normal Tariff)",
            "Taxi 1hour Waiting (Normal Tariff)",
            "Gasoline (1 liter)"]
            
dataMean = DevelopmentMean[column]
dataMedian = DevelopmentMedian[column]
arangeSize = len(column)
plt.figure(figsize = (30, 10))

plt.subplot(1, 2, 1)
plt.bar(np.arange(arangeSize) + 0.00, dataMean.iloc[0].values, color = "#004c6d", width = 0.25)
plt.bar(np.arange(arangeSize) + 0.25, dataMean.iloc[1].values, color = "#5385a8", width = 0.25)
plt.bar(np.arange(arangeSize) + 0.50, dataMean.iloc[2].values, color = "#95c4e8", width = 0.25)
plt.xticks([r + 0.25 for r in range(len(column))], column, fontsize = 15)
plt.xticks(rotation = 90)
plt.legend(["Undeveloped", "Underdeveloped", "Developed"], fontsize = 14)
plt.ylabel("Value", fontsize = 14)
plt.title("Mean of the transportation values", fontsize = 14)

plt.subplot(1, 2, 2)
plt.bar(np.arange(arangeSize) + 0.00, dataMedian.iloc[0].values, color = "#488f31", width = 0.25)
plt.bar(np.arange(arangeSize) + 0.25, dataMedian.iloc[1].values, color = "#75a760", width = 0.25)
plt.bar(np.arange(arangeSize) + 0.50, dataMedian.iloc[2].values, color = "#9fc08f", width = 0.25)
plt.xticks([r + 0.25 for r in range(len(column))], column, fontsize = 15)
plt.xticks(rotation = 90)
plt.legend(["Undeveloped", "Underdeveloped", "Developed"], fontsize = 14)
plt.ylabel("Value", fontsize = 14)
plt.title("Median of the transportation values", fontsize = 14)



### Utilities (Monthly)
x = Data["Basic (Electricity, Heating, Cooling, Water, Garbage) for 85m2 Apartment"]
y = Data["Internet (60 Mbps or More, Unlimited Data, Cable/ADSL)"]
w = Data["Development"]

plt.figure(figsize = (18, 10))
scatter = plt.scatter(x, y, c = w)
plt.xlabel("Basic (Electricity, Heating, Cooling, Water, Garbage) for 85m2 Apartment", fontsize = 15)
plt.ylabel("Internet (60 Mbps or More, Unlimited Data, Cable/ADSL)", fontsize = 15)
plt.title("Utilities (Monthly)", fontsize = 17)
legend = ["Undeveloped", "Underdeveloped", "Developed"]
plt.legend(handles = scatter.legend_elements()[0], labels = legend)



### Sports And Leisure
x = Data["Fitness Club, Monthly Fee for 1 Adult"]
y = Data["Cinema, International Release, 1 Seat"]
w = Data["Development"]

plt.figure(figsize = (18, 10))
scatter = plt.scatter(x, y, c = w)
plt.xlabel("Basic (Electricity, Heating, Cooling, Water, Garbage) for 85m2 Apartment", fontsize = 15)
plt.ylabel("Internet (60 Mbps or More, Unlimited Data, Cable/ADSL)", fontsize = 15)
plt.title("Utilities (Monthly)", fontsize = 17)
legend = ["Undeveloped", "Underdeveloped", "Developed"]
plt.legend(handles = scatter.legend_elements()[0], labels = legend)

    
        
        
### Childcare     
x = Data["Preschool (or Kindergarten), Full Day, Private, Monthly for 1 Child"] 
y = Data["International Primary School, Yearly for 1 Child"]
w = Data["Development"]


plt.figure(figsize=(20,12))
scatter = plt.scatter(x, y, c = w)
plt.xlabel("Preschool (or Kindergarten), Full Day, Private, Monthly for 1 Child", fontsize = 15)
plt.ylabel("International Primary School, Yearly for 1 Child", fontsize = 15)
plt.title("Childcare", fontsize = 17)
legend = ["Undeveloped", "Underdeveloped", "Developed"]
plt.legend(handles = scatter.legend_elements()[0], labels = legend)    
        
        
        
        
### Clothing And Shoes   
column = ["1 Pair of Jeans (Levis 501 Or Similar)",
        "1 Summer Dress in a Chain Store (Zara, H&M, ...)",
        "1 Pair of Nike Running Shoes (Mid-Range)",
        "1 Pair of Men Leather Business Shoes"]
            
dataMean = DevelopmentMean[column]
dataMedian = DevelopmentMedian[column]
arangeSize = len(column)
plt.figure(figsize = (30, 10))

plt.subplot(1, 2, 1)
plt.bar(np.arange(arangeSize) + 0.00, dataMean.iloc[0].values, color = "#004c6d", width = 0.25)
plt.bar(np.arange(arangeSize) + 0.25, dataMean.iloc[1].values, color = "#5385a8", width = 0.25)
plt.bar(np.arange(arangeSize) + 0.50, dataMean.iloc[2].values, color = "#95c4e8", width = 0.25)
plt.xticks([r + 0.25 for r in range(len(column))], column, fontsize = 15)
plt.xticks(rotation = 90)
plt.legend(["Undeveloped", "Underdeveloped", "Developed"], fontsize = 14)
plt.ylabel("Value", fontsize = 14)
plt.title("Mean of the clothing and shoes values", fontsize = 14)

plt.subplot(1, 2, 2)
plt.bar(np.arange(arangeSize) + 0.00, dataMedian.iloc[0].values, color = "#488f31", width = 0.25)
plt.bar(np.arange(arangeSize) + 0.25, dataMedian.iloc[1].values, color = "#75a760", width = 0.25)
plt.bar(np.arange(arangeSize) + 0.50, dataMedian.iloc[2].values, color = "#9fc08f", width = 0.25)
plt.xticks([r + 0.25 for r in range(len(column))], column, fontsize = 15)
plt.xticks(rotation = 90)
plt.legend(["Undeveloped", "Underdeveloped", "Developed"], fontsize = 14)
plt.ylabel("Value", fontsize = 14)
plt.title("Median of the clothing and shoes values", fontsize = 14)    
        
        
        
        
### Rent Per Month    
column = ["Apartment (1 bedroom) in City Centre",
        "Apartment (1 bedroom) Outside of Centre",
        "Apartment (3 bedrooms) in City Centre",
        "Apartment (3 bedrooms) Outside of Centre"]
            
dataMean = DevelopmentMean[column]
dataMedian = DevelopmentMedian[column]
arangeSize = len(column)
plt.figure(figsize = (30, 10))

plt.subplot(1, 2, 1)
plt.bar(np.arange(arangeSize) + 0.00, dataMean.iloc[0].values, color = "#004c6d", width = 0.25)
plt.bar(np.arange(arangeSize) + 0.25, dataMean.iloc[1].values, color = "#5385a8", width = 0.25)
plt.bar(np.arange(arangeSize) + 0.50, dataMean.iloc[2].values, color = "#95c4e8", width = 0.25)
plt.xticks([r + 0.25 for r in range(len(column))], column, fontsize = 15)
plt.xticks(rotation = 90)
plt.legend(["Undeveloped", "Underdeveloped", "Developed"], fontsize = 14)
plt.ylabel("Value", fontsize = 14)
plt.title("Mean of the rent per month values", fontsize = 14)

plt.subplot(1, 2, 2)
plt.bar(np.arange(arangeSize) + 0.00, dataMedian.iloc[0].values, color = "#488f31", width = 0.25)
plt.bar(np.arange(arangeSize) + 0.25, dataMedian.iloc[1].values, color = "#75a760", width = 0.25)
plt.bar(np.arange(arangeSize) + 0.50, dataMedian.iloc[2].values, color = "#9fc08f", width = 0.25)
plt.xticks([r + 0.25 for r in range(len(column))], column, fontsize = 15)
plt.xticks(rotation = 90)
plt.legend(["Undeveloped", "Underdeveloped", "Developed"], fontsize = 14)
plt.ylabel("Value", fontsize = 14)
plt.title("Median of the rent per month values", fontsize = 14)    




### Buy Apartment Price
x = Data["Price per Square Meter to Buy Apartment in City Centre"] 
y = Data["Price per Square Meter to Buy Apartment Outside of Centre"]
w = Data["Development"]


plt.figure(figsize=(20,12))
scatter = plt.scatter(x, y, c = w)
plt.xlabel("Price per Square Meter to Buy Apartment in City Centre", fontsize = 15)
plt.ylabel("Price per Square Meter to Buy Apartment Outside of Centre", fontsize = 15)
plt.title("Buy Apartment Price", fontsize = 17)
legend = ["Undeveloped", "Underdeveloped", "Developed"]
plt.legend(handles = scatter.legend_elements()[0], labels = legend)




### Salaries and Financing
x = Data["Average Monthly Net Salary (After Tax)"] 
y = Data["Mortgage Interest Rate in Percentages (%), Yearly, for 20 Years Fixed-Rate"]
z = Data["Development"]


plt.figure(figsize=(20,12))
scatter = plt.scatter(x, y, c = z)
plt.xlabel("Average Monthly Net Salary (After Tax)", fontsize = 15)
plt.ylabel("Mortgage Interest Rate in Percentages (%), Yearly, for 20 Years Fixed-Rate", fontsize = 15)
plt.title("Salaries and Financing", fontsize = 17)
legend = ["Undeveloped", "Underdeveloped", "Developed"]
plt.legend(handles = scatter.legend_elements()[0], labels = legend)



## Machine Learning
X = Data.iloc[:, 1:-1]
Y = Data.iloc[:, -1:]


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.33, random_state = 0)


### Random Forest
RF = RandomForestClassifier(max_depth=5, random_state=0)
RF.fit(X_train, y_train.values.ravel())
predRF = RF.predict(X_test)

RFscore = metrics.accuracy_score(y_test, predRF)
print("Random Forest Classifier accuracy score : {}".format(RFscore))

cvRF = cross_val_score(RF, X, Y.values.ravel(), cv = 8, scoring='accuracy')
print("Random Forest Classifier CVS score      : {}".format(cvRF.mean()))



### Decision Tree Classifier
DT = DecisionTreeClassifier(max_depth = 3, random_state = 0)  
DT.fit(X_train, y_train.values.ravel())
predDT = DT.predict(X_test)

DTscore = metrics.accuracy_score(y_test, predDT)
print("Decision Tree Classifier accuracy score : {}".format(DTscore))

cvDT = cross_val_score(DT, X, Y.values.ravel(), cv = 8, scoring='accuracy')
print("Decision Tree Classifier CVS score      : {}".format(cvDT.mean()))



### Support Vector Machines (SVM)
SVM = svm.SVC()
SVM.fit(X_train, y_train.values.ravel())
predSVM = SVM.predict(X_test)

SVMscore = metrics.accuracy_score(y_test, predDT)
print("Support Vector Machines accuracy score : {}".format(SVMscore))

cvSVM = cross_val_score(SVM, X, Y.values.ravel(), cv = 8, scoring='accuracy')
print("Support Vector Machines CVS score      : {}".format(cvSVM.mean()))




### GaussianNB
GaNB = GaussianNB()
GaNB.fit(X_train, y_train.values.ravel())
predGaNB = GaNB.predict(X_test)

GaNBscore = metrics.accuracy_score(y_test, predGaNB)
print("GaussianNB accuracy score : {}".format(GaNBscore))

cvGaNB = cross_val_score(GaNB, X, Y.values.ravel(), cv = 8, scoring='accuracy')
print("GaussianNB CVS score      : {}".format(cvGaNB.mean()))



### K-Neighbors Classifier (KNN)
KNN = KNeighborsClassifier(n_neighbors=3)
KNN.fit(X_train, y_train.values.ravel())
predKNN = KNN.predict(X_test)

KNNscore = metrics.accuracy_score(y_test, predKNN)
print("K-Neighbors Classifier accuracy score : {}".format(KNNscore))

cvKNN = cross_val_score(GaNB, X, Y.values.ravel(), cv=8, scoring='accuracy')
print("K-Neighbors Classifier CVS score      : {}".format(cvKNN.mean()))


















































        
        
        
        
        
        