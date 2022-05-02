import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import warnings
warnings.filterwarnings('ignore')

class acc:
  def accuracyScore(self,x_train, y_train, x_test, y_test, classifier):
      classifier.fit(x_train, y_train)
      y_pred_train = classifier.predict(x_train) #this will gives if the training dataset is spam or not spam
      y_pred_test = classifier.predict(x_test) #this will gives if the testing dataset is spam or not spam
      return accuracy_score(y_train, y_pred_train),accuracy_score(y_test, y_pred_test)


if __name__ == "__main__":
  data = pd.read_csv("spambase.csv",header=None) #copying the whole data from csv file to a variable named data
  x = data.drop([57],axis='columns') #getting all columns except the last column in data
  y = data.iloc[:,-1] #getting last column in data
  x = StandardScaler().fit_transform(x)
  x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3) #train_test_split function is used to split the data based on the given test_size#C values for regulization
  C_values = [0.00005, 0.0005, 0.005, 0.05, 0.5, 1, 5,50,500,5000] #C values for classification
  AccTable = pd.DataFrame(columns=['C', 'Linear', 'Poly', 'RBF']) #Different type of kernels
  obj=acc()
  for eachC_value in C_values:
    accuracies = []
    for ker in ['linear', 'poly', 'rbf']:
      classifier = SVC(C=eachC_value, kernel=ker , max_iter=1e8) 
      accuracy = obj.accuracyScore(x_train, y_train, x_test, y_test, classifier)
      accuracies.append(accuracy)
    AccTable = AccTable.append({'C': eachC_value, 'Linear': accuracies[0], 'Poly': accuracies[1], 'RBF': accuracies[2]}, ignore_index=True)

  print(AccTable)