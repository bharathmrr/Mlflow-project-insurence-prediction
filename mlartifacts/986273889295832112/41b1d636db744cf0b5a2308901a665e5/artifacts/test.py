import mlflow
print('printing tracking uri')
print(mlflow.get_tracking_uri())
print('\n')
mlflow.set_tracking_uri("http://localhost:5000")
print('printing tracking uri after setting')
print(mlflow.get_tracking_uri())
print('\n')

