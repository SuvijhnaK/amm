import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

PlayTennis = pd.read_csv("4.csv")
Le= LabelEncoder()

print("Given Dataset\n",PlayTennis,"\n")

PlayTennis["outlook"]= Le.fit_transform(PlayTennis["outlook"])
PlayTennis['temp'] = Le.fit_transform(PlayTennis['temp'])
PlayTennis['humidity'] = Le.fit_transform(PlayTennis['humidity'])
PlayTennis['wind'] = Le.fit_transform(PlayTennis['wind'])
PlayTennis['play'] = Le.fit_transform(PlayTennis['play'])

print("Encoded Dataset is :\n",PlayTennis,"\n")

X=PlayTennis.drop(['play'],axis=1)
y=PlayTennis['play']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20)

print("\n X_train:\n",X_train)
print("\n y_train:\n",y_train)
print("\n X_test:\n",X_test)
print("\n y_test:\n",y_test)

classifier = GaussianNB()
classifier.fit(X_train,y_train)

accuracy = accuracy_score(classifier.predict(X_test),y_test)
print("\n Accuracy is:",accuracy)
