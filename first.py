import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



df=pd.read_csv("mail_data.csv")
print(df)



data=df.where((pd.notnull(df)),'')


data.head()#top rows of data set


data.info()


data.shape(5572,2)


data.loc[data['Category']=='spam','Category',]=0
data.loc[data['Category']=='ham','Category',]=1


X=data['Message']
Y=data['Category']


print(X)

print(Y)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=3)
print(X.shape)
print(X_train)
print(X_test.shape)



print(Y.shape)
print(Y_train)
print(Y_test.shape)


feature_extraction=TfidfVectorizer(min_df=1,stop_words="english",lowercase='True')
X_train_features=features_extraction.fit_transform(X_train)
X_test_features= feature_extraction.transform(X_test)

Y_train= Y_train.astype('int')
Y_test= Y_test.astype('int')

print(X_train)

print(X_train_features)


model=LogisticRegression()

model.fit(X_train_features,Y_train)


prediction_on_training_data=model.predict(X_train_features)
accuracy_on_training_data=accuracy_score(Y_train,prediction_on_training_data)


print('ACCURACY ON TRAINING DATA:',accuracy_on_training_data)

prediction_on_test_data= model.predict(X_test_features)
accuracy_on_test_data=accuracy_score(Y_test,prediction_on_test_data)

print('accuracy on test data:',accuracy_on_test_data)

input_your_mail= [""]
input_data_features= feature_extraction.transform(input_your_mail)
prediction=model.predict(input_data_features)
print(prediction)
if prediction[0]==1:
    print('handwritten email')
else:
    print("spam email")
