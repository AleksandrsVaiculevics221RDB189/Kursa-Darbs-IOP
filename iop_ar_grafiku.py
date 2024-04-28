#importesim math priekš fukcijas ievadīšanai
import math
# numpy vajadzigs gradienta funkcijai
import numpy as np
'''
kā trešo algoritmu importēsim minimize no scipy bibliotēkas
tā ir bibliotēka, kura ir iebuvētās dažadas optimizācijas algoritmi
mēs, šajā gadijumā izmantosim minimize funkciju, jo par to bija viegāk
atrast informāciju internetā
'''
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# definesim funkciju
# x1**2  tas pats ka x1^2
def pamata_funkcija (a, b, c, d, l, k, x1, x2, x3):
    funkcija = ( a*x1**4 ) - ( b*x1*x2**2 ) + ( c*x2**2*x3**2) \
     - ( d*x3**3 ) + (l*x1) - (k*x2) + (math.exp(x3)) - math.log(x1**2 + x2**2 + 1)
    return funkcija

# pec 9. varianta izmainisim mainigos
a=-2
b=10
c=-4
d=10
l=10
k=-4

#==============================================================================
# Gradient algoritms izmantojot studiju materialus
#==============================================================================
# https://medium.com/@ronaktali/numerical-algorithms-gradient-descent-and-newtons-method-ba7256e4d8c2
# https://omaraflak.medium.com/optimization-descent-algorithms-bf595f069788
#==============================================================================

'''''
# gradienta algoritma pamata ir formula:
# f(p) = |df/dx1 (p) |
#        |......     |
#        |df/dxn (p) | 
# lai izmantot formulu, mums vajag tris parciālie atvasinājumi pec x1,x2,x3, un izmantosim tos ka array :
# lai noteiktu atvasinajumus, tiek izmantots photomath rīks
'''''

def parciālie_atvasinājumi(x1, x2, x3):
    dx1 = -((2*x1)/(x1**2+x2**2+1)) + (4*a*x1**3) - (b*x2**2) + l
    dx2 = -((2*x2)/(x2**2+x1**2+1)) + (2*c*x3**2 * x2) - (2*b*x1*x2) - k
    dx3 = math.exp(x3)-(3*d*x3**2) + 2*c*x2**2*x3
    return np.array([dx1, dx2, dx3])


'''''
  talak mums vajag izmantot formulu "Gradient direction"
  d = -alfa * deltax * f(x)
  kura ir daļa no lielākas formulas,
  x(k+1) = xk - -alfa * deltax * f(xl)
  mēs šeit samainīsim alfa uz t
  šeit alfa ir soļa garums, step size t, kurš ir numurs starp [0,1]
  epsilon ir precizitates limenis
  t un epsilon var mainīt pēc uzdevuma nosacījumiem:
'''''

# šis ir tikai algoritms talak to izmanto optimuma apreķināšanai
def Gradienta_algortims(gradientu_aprek, x1x2x3_grad, t=0.00001, epsilon=1e-10, max_iterations=1000):
    x = x1x2x3_grad
    trajectory = [x]  # Initialize the trajectory with the initial point
    for i in range(max_iterations):
        x = x - t * gradientu_aprek(*x)
        trajectory.append(x)  # Append the new point to the trajectory
        if np.linalg.norm(gradientu_aprek(*x)) < epsilon:
            return x, trajectory
    return x, trajectory



# x1 var but starp -0.7, 0,000... līdz 1.2
# x2 var but starp -1.6, 0,000... līdz 1.8
# x3 var but starp -10... 0.000... līdz 0.9
x1x2x3_grad = (0.1, 0.1, 0.1)  # šeit var mainīt mainīgos x1,x2 un x3

'''
šeit ar lambda funkciju palidzibu mes izmantosim x1,x2,x3 kurus paņem no
parcialiem atvasinajumiem, un to visu izmantosim caur gradienta algoritmu
šeit ari var mainīt iterāciju skaitu un soļa garumu t
'''

# Adjust the function call to receive only two values
x1_min, trajectory = Gradienta_algortims(lambda x1, x2, x3: parciālie_atvasinājumi(x1, x2, x3), x1x2x3_grad, t=0.00001, max_iterations=5000)

# Remove the unpacking of 'it' since the function returns only two values
print ('==============================================================================')
print ('Gradienta optimizācija')
print ('==============================================================================')
# Print the shape and content of the trajectory array
print('Length of trajectory array:', len(trajectory))
print('Content of trajectory array:', trajectory)



#print('Optimizētās/minimizētās vērtības : ', x1x2x3_min_grad)
#print('Pamata funkcijas vertībā pie opt.vert =', pamata_funkcija(a, b, c, d, l, k, *x1x2x3_min_grad))
#print('p. f. v. pēc gradienta alg. pie opt.vert =', parciālie_atvasinājumi(*x1x2x3_min_grad))
trajectory = np.array(trajectory)
plt.figure(figsize=(10, 6))
plt.plot(trajectory[:, 0], label='x1')
plt.plot(trajectory[:, 1], label='x2')
plt.plot(trajectory[:, 2], label='x3')
plt.xlabel('Итерация')
plt.ylabel('Значение переменной')
plt.title('Траектория градиентного спуска')
plt.legend()
plt.show()