import numpy as np

def mae(y_true, y_pred):
  #without np
  total_error = 0
  for yt, yp in zip(y_true, y_pred):
    total_error += abs(yt-yp)
  print("Total error: ", total_error)
  mae = total_error/len(y_true)
  print("Mean absolute error: ", mae)

  #using numpy
  return np.mean(np.abs(y_true-y_pred))

def mse(y_true, y_pred):
  #without numpy
  total_error = 0
  for yt, yp in zip(y_true, y_pred):
    total_error += (yt-yp)**2
  print("Total error: ", total_error)
  mse = total_error/len(y_true)
  print("Mean squared error: ", mse)

  #using numpy
  return np.mean(np.square(y_true-y_pred))

def logloss(y_true, y_pred):
  '''
  the reason we are converting y_prediction values is because log(0) is undefined 
  suppose, predicited value = 0 ==> log(y_pred) --> undefined. 
  Therfore, we convert all 0s to value close to zero (1e-15) - min(i,epsilon)
  suppose, predicted = 1 and true = 1 ==> log(1-1) -->unndefined. 
  Therfore, we convert all 1s to value close to 1 (1-epsilon) - max(1-i,epsilon)
  '''
  epsilon = 1e-15
  y_pred_new = [max(i, epsilon) for i in y_pred]
  y_pred_new = [min(i, 1-epsilon) for i in y_pred_new]
  y_pred_new = np.array(y_pred_new)
  return -np.mean(y_true*np.log(y_pred_new)+(1-y_true)*np.log(1-y_pred_new))

y_true = np.array([0.30, 0.7, 1, 0, 0.5])
y_pred = np.array([1, 1, 0, 0, 1])

print(f"Mean absolute error = {mae(y_true, y_pred)}")
print(f"Mean squared error = {mse(y_true, y_pred)}")
print(f"Binary crossentropy = {logloss(y_true, y_pred)}")
