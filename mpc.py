import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

""" Variables que se pueden cambiar """

A = np.array([[0,1],[0.5,0.44]])
B= np.array([[0],[0.6]])
C = np.array([[0,1]])

Q = np.diag([0.2])
S = np.diag([0.3])
R = np.diag([0.4])

Hz = 3

""" Fin de variables que se pueden cambiar """


""" Poner todo esto en una función """
A_aug = np.hstack([A,B])
temp1=np.zeros((np.size(B,1),np.size(A,1)))
temp2=np.identity(np.size(B,1))
temp=np.hstack([temp1,temp2])

A_aug = np.vstack([A_aug,temp])
B_aug = np.vstack([B,np.identity(np.size(B,1))])
C_aug = np.hstack([C,np.zeros([np.size(C,0),np.size(B,1)])])

CQC = C_aug.T@Q@C_aug
CSC = C_aug.T@S@C_aug
CQC_arr = [CQC]*(Hz)
CQC_arr[-1]=CSC

QC = Q@C_aug 
SC = S@C_aug
QC_arr = [QC]*(Hz)
QC_arr[-1] = SC

R_arr = [R]*Hz

Qdb = block_diag(*CQC_arr)
Tdb = block_diag(*QC_arr)
Rdb = block_diag(*R_arr)

Cdb = block_diag(*([B_aug]*Hz))

for row in range(1,Hz):
    for column in range(0,Hz):
        if np.array_equal(B_aug.T[0],Cdb[ row*(np.size(B,0)+1): (np.size(B,0)+1)*2*row , column]):
            break
        Cdb[ row*(np.size(B,0)+1): (np.size(B,0)+1)*row+(np.size(B,0)+1) , column] = ( np.linalg.matrix_power(A_aug,row-column)@B_aug ).T 



Adc = np.zeros([np.size(A_aug,0)*Hz,np.size(A_aug,1)])

for row in range(0,Hz):
    Adc[row*np.size(A_aug,0): np.size(A_aug,0)*(row+1) ,:] = np.linalg.matrix_power(A_aug,row+1)

Hdb = Cdb.T@Qdb@Cdb + Rdb

temp  = Adc.T@Qdb@Cdb
temp1 = -Tdb@Cdb

Fdbt = np.vstack([temp,temp1])   
""" Poner todo esto en una función Lo importante es Hdb y Fdbt """

states = np.array([[0,0]])
ref = np.ones((Hz,1))*3
U = np.array([[0]])

sal = np.zeros(20)

for k in range(0,20):

    states_aug_t = np.hstack([states,U]).T

    aux = np.vstack([states_aug_t,ref]).T

    Ft = aux@Fdbt

    du = -np.linalg.inv(Hdb)@Ft.T

    U = U + du[0,0]

    states = A@states.T + B*U 
    Y = C@states
    states = states.T

    sal[k] = Y

plt.plot(sal)
plt.grid()
plt.title("MPC sistema masa amortiguador resorte")
plt.ylabel("Posición [m]")
plt.xlabel("Time")
plt.show()
