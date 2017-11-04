import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import StandardScaler

wine = pd.read_csv('C:/Users/ANJALI/Desktop/wine_dataset.csv', names = ["Cultivator", "Alchol", "Malic_Acid", "Ash", "Alcalinity_of_Ash", "Magnesium", "Total_phenols", "Falvanoids", "Nonflavanoid_phenols", "Proanthocyanins", "Color_intensity", "Hue", "OD280", "Proline"])
wine.head()
wine.describe().transpose()
wine.shape
X = wine.drop('Cultivator',axis=1)
y = wine['Cultivator']

X_train, X_test, y_train, y_test = train_test_split(X, y)

scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)
# Now apply the transformations to the data:
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
mlp = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500)
mlp.fit(X_train,y_train)
predictions = mlp.predict(X_test)
print("Accuracy = ",accuracy_score(y_test, predictions)*100, "%")
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
''' coefs_ is a list of weight matrices, where weight matrix at index i represents the weights between layer i and layer i+1.
    intercepts_ is a list of bias vectors, where the vector at index i represents the bias values added to layer i+1.'''
#print the following to check the value of coefficients and intercepts
'''
print(len(mlp.coefs_))
print(len(mlp.coefs_[0]))
print(len(mlp.intercepts_[0]))'''