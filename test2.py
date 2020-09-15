import os
import time

t = time.time()
os.system("python sleep5s.py &")
os.system("python sleep5s.py &") 
os.system("python sleep5s.py &")
print(time.time() - t)



