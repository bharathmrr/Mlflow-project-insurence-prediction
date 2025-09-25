import mlflow 
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Iris_Classification")
data= load_iris()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
n_estimators = 10
random_state = 42
max_depth = 10
with mlflow.start_run():
   rf= RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, max_depth=max_depth)
   rf.fit(X_train, y_train) 
   y_pred = rf.predict(X_test)

   accuracy = accuracy_score(y_test, y_pred)
   cm = confusion_matrix(y_test, y_pred)

   mlflow.log_param("n_estimators", n_estimators)
   mlflow.log_param("random_state", random_state)
   mlflow.log_param("max_depth", max_depth)
   mlflow.log_metric("accuracy", accuracy)
   plt.figure(figsize=(8,6))
   sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=data.target_names, yticklabels=data.target_names)
   plt.xlabel('Predicted')   
   plt.ylabel('True')
   plt.title('Confusion Matrix')
   plt.savefig('confusion_matrix.png')
   plt.show()
   mlflow.log_artifact('confusion_matrix.png')
   mlflow.log_artifact('src/test.py')
   mlflow.log_artifact(__file__)
   mlflow.sklearn.log_model(rf, "model")
   mlflow.set_tags({"developer":"bharath","model":"RandomForestClassifier"})
   print(f"Model accuracy: {accuracy}")
    