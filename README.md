## EX:05 Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student
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

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SRI HARI KRISHNA D T
RegisterNumber: 212224240160
*/

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Removes the specified row or column
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
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")# A library for large linear classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)# Accuracy Score = (TP+TN)/
#accuracy_score(y_true,y_prednormalize=False)
accuracy

from sklearn.metrics import confusion_matrix
confusion = (y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
1.PLACEMENT DATA:

![image](https://github.com/user-attachments/assets/0b5c60f5-9f80-47a8-af93-63bc7cfc3bf7)

2.SALARY DATA:

![image](https://github.com/user-attachments/assets/ff8c96d9-3d7b-4b20-bd2b-0b7e9c76a6a3)

3.CHECKING THE NULL() FUNCTION:

![image](https://github.com/user-attachments/assets/03a4afdf-af3e-4917-a851-b13f3612f6e3)


4.DATA DUPLICATE:

![image](https://github.com/user-attachments/assets/aef28896-66a4-45f8-9627-58073e5e0d28)


5.PRINT DATA:

![image](https://github.com/user-attachments/assets/c7ed051e-1bdd-4ec3-a7de-84d65f420118)


6.DATA STATUS:

![image](https://github.com/user-attachments/assets/6ce416f2-fd2a-45c8-8c53-e75db4ed2dc0)


![image](https://github.com/user-attachments/assets/4632993d-74e8-4d53-89cb-fca7b4cf3c6b)

7.Y_PREDICATION ARRAY:

![image](https://github.com/user-attachments/assets/9a5e8869-9aae-4f5d-b050-61fd0605ad68)


8.ACCURACY VALUE:


![image](https://github.com/user-attachments/assets/80151e71-a2f3-4c0a-ab6a-c141ed527a27)



9.CONFUSION ARRAY:

![image](https://github.com/user-attachments/assets/c05f6ba3-2f36-40ed-95f7-751bbd98b869)

![image](https://github.com/user-attachments/assets/26c6ff10-d363-42dd-8464-14a086439366)

10.CLASSIFICATION REPORT:

![image](https://github.com/user-attachments/assets/d75d16b4-9d55-43d2-a8af-dfbe9ab6962d)


PREDICTION OF LR:

![image](https://github.com/user-attachments/assets/06825227-8364-4518-84d3-5c5044379688)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
