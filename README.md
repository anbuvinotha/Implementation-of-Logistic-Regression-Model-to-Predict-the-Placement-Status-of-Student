# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results

## Program:

/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: ANBU VINOTHA.S
RegisterNumber:  212223230015
*/
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1

x=data1.iloc[:,:-1]
x
y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


## Output:

## DATA HEAD
![Screenshot 2025-03-27 144229](https://github.com/user-attachments/assets/f1f948cd-7a83-46ac-bc24-eb4d1ca10bbb)

## DATA1 HEAD
![Screenshot 2025-03-27 144307](https://github.com/user-attachments/assets/1aa2c32b-f6f2-4fa9-8286-24e18d190fdc)

## ISNULL().SUM()
![Screenshot 2025-03-27 144341](https://github.com/user-attachments/assets/9499215e-6483-4041-9bb6-ab8f96cdccba)

## DATA DUPLICATE
![Screenshot 2025-03-27 144358](https://github.com/user-attachments/assets/c7387294-2a85-40e8-aaf6-7cbecdba7232)

## PRINT DATA
![Screenshot 2025-03-27 144419](https://github.com/user-attachments/assets/97a37dfb-5e22-47ac-a358-a3c0674842ff)

## STATUS
![Screenshot 2025-03-27 144434](https://github.com/user-attachments/assets/0972b01c-40fc-4f97-9e06-b189f0e8a8e1)

## Y_PRED
![Screenshot 2025-03-27 144451](https://github.com/user-attachments/assets/716c6219-7077-4115-99ff-6371608feb53)

## ACCURACY
![Screenshot 2025-03-27 144506](https://github.com/user-attachments/assets/c03dbc91-7981-42bd-b68c-cb618d8aaae9)

## CONFUSION MATRIX
![Screenshot 2025-03-27 144521](https://github.com/user-attachments/assets/6068d2d9-ee59-4cf5-bf58-7ba07b351fad)

## CLASSIFICATION
![Screenshot 2025-03-27 144534](https://github.com/user-attachments/assets/8d0529ac-9d5f-4fa6-9ce5-6f23bced3436)

## LR PREDICT
![Screenshot 2025-03-27 144559](https://github.com/user-attachments/assets/e73d7d65-82f3-4c48-886d-235a17a90913)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
