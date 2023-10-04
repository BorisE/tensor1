import numpy as np

import os
os.system("")
CHEAD = '\033[92m'
CRED = '\033[91m'
CYEL = '\033[33m'
CWRONG = '\033[2;31;43m'
CEND = '\033[0m'

# https://stackoverflow.com/questions/1589706/iterating-over-arbitrary-dimension-of-numpy-array

arr = np.arange(36).reshape(3,3,1,4)
#arr = np.array([5,5,5,5,3,3,3,3,2,2,2,2,1,1,1,1,7,7,7,7,8,8,8,8]).reshape(2,3,4)

print (arr.shape)
print (arr)
print ("Last dimension size: %d" % (arr.shape[-1]))


for i in range(arr.shape[-1]):
    print ("Line:")
    print (arr[:,:,0,i])


loss=0.0232323
accuracy = 0.99956

print("Test loss: %.3f" % loss)
print("Test loss: %.3f" % loss + ", test accuracy: " + CYEL + "%.2f%%" % (accuracy*100.0) + CEND )
print(("Test loss: " + CYEL + "%.3f" + CEND + ", test accuracy: " + CYEL + "%.2f%%"+ CEND )% (loss, accuracy*100.0)  )
