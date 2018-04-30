import pandas as pd
import time
train=pd.read_csv('new_train.csv',encoding='latin-1')

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score
start_time=time.time()
train.dropna(inplace=True)
#languages=['en','fr','de']
#train=train.loc[train['lang'].isin(languages)]
#print(train.shape)
print('Splitting Data......................')
train_data,val_data=train_test_split(train,test_size=0.2,random_state=123)
le=LabelEncoder()
vec=TfidfVectorizer(max_features=50000)

train_text=train_data['text'].values
val_text=val_data['text']
#Data Preprocessing
print('Data Preprocessing...........................')
X_train=vec.fit_transform(train_text)
X_val=vec.transform(val_text)
y_train=le.fit_transform(train_data['lang'])
y_val=le.transform(val_data['lang'])
print('Model Building...................')
#Model Building
model=LogisticRegression()
model.fit(X_train,y_train)
print('Model Prediction and scoring.............')
#Model Prediction
y_pred=model.predict(X_val)
print(precision_score(y_val,y_pred,average='macro'))
total_time=time.time()-start_time
print('time taken'+ str(total_time))


def predictSentence(sentence):
    pred=model.predict(vec.transform([sentence]))
    return le.classes_[pred]