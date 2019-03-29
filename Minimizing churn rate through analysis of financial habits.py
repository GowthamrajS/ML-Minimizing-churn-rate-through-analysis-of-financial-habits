# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea

dataset = pd.read_csv("churn_data.csv")

dataset.isnull().sum()

dataset = dataset.drop(["rewards_earned","credit_score"],axis = True)

dataset1 = dataset["age"].median()
dataset["age"] = dataset["age"].fillna(dataset1)

dataset.isnull().sum()

"""
dataset2 =dataset.copy().drop(["user","churn","zodiac_sign","housing","payment_type"],axis =1)

plt.figure(figsize = (10,10))
plt.suptitle("Histgram",fontsize = 20)

for i in range(1,dataset2.shape[1]+1):
    plt.subplot(6,4,i)
    f = plt.gca()
    f.axes.get_yaxis().set_visible(False)
    f.set_title(dataset2.columns.values[i-1])

    value =np.size(dataset2.iloc[:,i-1].unique())
    plt.hist(dataset2.iloc[:,i-1],bins = value,color = "#3F5D7D")
plt.tight_layout(rect =[0,0.06,1,0.95])


"""


dataset = pd.get_dummies(dataset)


y = dataset["churn"]

dataset = dataset.drop(["housing_na","payment_type_na","zodiac_sign_na","user","churn"],axis = 1)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(dataset,y,train_size = 0.8)

from sklearn.preprocessing import StandardScaler as scaler

scaler = scaler()

x_train = scaler.fit_transform(x_train)

x_test = scaler.transform(x_test)

from sklearn.ensemble import RandomForestClassifier as rf

rf = rf()

rf.fit(x_train,y_train)

y_p = rf.predict(x_test)

from sklearn.metrics import confusion_matrix , accuracy_score

ac = accuracy_score(y_test,y_p)

cm = confusion_matrix(y_test,y_p)
sea.heatmap(cm,annot = True,fmt="g")









