# CFD Basic Python Programmming
# (a) Basic Math Operation 

import numpy as np
N = 10
lambda_ = 0.8


M_ones = np.diag(np.full(N,1))
M_r = np.diag(np.ones(N-1),1)*lambda_/2
M_l = np.diag(np.ones(N-1),-1)*(-lambda_/2)
Mfull = M_ones + M_r + M_l
Mfull[0,N-1] = -lambda_/2
Mfull[N-1,0] = lambda_/2
Mfullinv = np.linalg.inv(Mfull)
print("Mfull = \n", Mfullinv)
