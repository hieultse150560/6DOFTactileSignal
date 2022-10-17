import pickle
import numpy as np
with open('./predictions/L2/singlePeople_pvtLarge_dis.p', 'rb') as f:
  dis = pickle.load(f)
  print("100 epochs: ")
  print(type(dis), dis.shape)
  full = np.mean(np.abs(dis), axis = 0)
  print(full)
  print("Ave: ", np.mean(full, axis = 0))
  print("Head: ", full[0])
  print("Shoulder: ", (full[2] + full[5])/2)
  print("Elbow: ", (full[3] + full[6])/2)
  print("Wrist: ", (full[4] + full[7])/2)
  print("Hip: ", full[8])
  print("Knee: ", (full[10] + full[13])/2)
  print("Ankle: ", (full[11] + full[14])/2)
  print("Feet: ", (full[15] + full[16] + full[18] + full[19])/4)
  
  print()
