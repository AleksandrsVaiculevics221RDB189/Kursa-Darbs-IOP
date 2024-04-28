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
    for i in range(max_iterations):
        x = x - t * gradientu_aprek(*x)
        if np.linalg.norm(gradientu_aprek(*x)) < epsilon:
            return x, i + 1
    return x, max_iterations

# x1 var but starp -0.7, 0,000... līdz 1.2
# x2 var but starp -1.6, 0,000... līdz 1.8
# x3 var but starp -10... 0.000... līdz 0.9
x1x2x3_grad = (0.1, 0.1, 0.1)  # šeit var mainīt mainīgos x1,x2 un x3

'''
šeit ar lambda funkciju palidzibu mes izmantosim x1,x2,x3 kurus paņem no
parcialiem atvasinajumiem, un to visu izmantosim caur gradienta algoritmu
šeit ari var mainīt iterāciju skaitu un soļa garumu t
'''
x1x2x3_min_grad, it = Gradienta_algortims(lambda x1, x2, x3: parciālie_atvasinājumi(x1, x2, x3), x1x2x3_grad, t=0.00001, max_iterations=5000)

print ('==============================================================================')
print ('Gradienta optimizācija')
print ('==============================================================================')
print('Iterāciju skaits =', it)
print('Izvelētās x1,x2,x3 vērtībās : ', x1x2x3_grad)
print('Optimizētās/minimizētās vērtības : ', x1x2x3_min_grad)
print('Pamata funkcijas vertībā pie opt.vert =', pamata_funkcija(a, b, c, d, l, k, *x1x2x3_min_grad))
#print('p. f. v. pēc gradienta alg. pie opt.vert =', parciālie_atvasinājumi(*x1x2x3_min_grad))


#==============================================================================
# Njutona algoritms izmantojot studiju materialus
# https://medium.com/@ronaktali/numerical-algorithms-gradient-descent-and-newtons-method-ba7256e4d8c2
# https://omaraflak.medium.com/optimization-descent-algorithms-bf595f069788
#==============================================================================
'''
Lai izveidotu Ņutona algoritmu, mums paša sakumā ir nepieciešama matrica 
no parc. atvasinajumiem, kuru mes ņemsim pēc hessian funkcijas formulas:
Hf = | d^2f/dx^2  d^2f/dxdy |
     | d^2f/dydx  d^2f/dy^2 |
     | ...                  |
kuru mes atrisināsim izmantojot saite ar atvasinajumu kalkulātoru
https://www.symbolab.com/solver/derivative-calculator
'''

def hessian_matrica_no_pamata_funkcijas(x1, x2, x3):
    d2_dx12 = (12*a*x1**2) - (2*(x2**2 + 1 - x1**2)/(x1**2+x2**2+1)**2)
    d2_dx1dx2 = ((4*x1*x2)/(x1**2+x2**2+1)**2) - 2*b*x2
    d2_dx1dx3 = 0
    d2_dx22 = (2*c*x3**2) - (2*(x1**2 + 1 - x2**2)/(x1**2+x2**2+1)**2)
    d2_dx2dx1 = ((4*x1*x2)/(x1**2+x2**2+1)**2) - 2*b*x2
    d2_dx2dx3 = 4*x2*c*x3
    d2_dx32 =  -6*d*x3 + 2*x2**2 * c + math.exp(x3)
    d2_dx3dx1 = 0
    d2_dx3dx2 = 4*x2*c*x3 
           
    return np.matrix([
        [d2_dx12, d2_dx1dx2, d2_dx1dx3],
        [d2_dx2dx1, d2_dx22, d2_dx2dx3],
        [d2_dx3dx2, d2_dx3dx1, d2_dx32]
    ])
'''
Mes izmantosim Newtons direction formulu,
d = -Hf * delta f(p)
bet izmainīsim to šada veidā: x = x - Hf*delta f(p)
lai to būtu vieglāk izmantot musu gadijumā
šeit var mainīt epsilon.

'''
def Nutona_algoritms(grad, hess, x1x2x3_newt, epsilon=1e-10, max_iterations=1000):
    x = x1x2x3_newt
    for i in range(max_iterations):
        x = x - np.linalg.solve(hess(*x), grad(*x))
        if np.linalg.norm(grad(*x)) < epsilon:
            return x, i + 1
    return x, max_iterations

# šeit var uzstātīt sakuma punktus
# strada ar vairākiem skaitļiem pat līdz -15, 15
x1x2x3_newt = (0.1, 0.1, 0.1)
'''
Šeit tika izsaukts ņutona algoritms, kura tiek izmantoti parc. atv, atvasinajumu matrica, sakuma punkti un maksimalas iesp. iterācijas
'''
x1x2x3_min_newt, it = Nutona_algoritms(parciālie_atvasinājumi, hessian_matrica_no_pamata_funkcijas, x1x2x3_newt, max_iterations=5000)

print ('==============================================================================')
print ('Ņutona optimizācija')
print ('==============================================================================')
print('Iterāciju skaits =', it)
print('Izvelētās x1,x2,x3 vērtībās : ', x1x2x3_newt)
print('Optimizētās/minimizētās vērtības : ', x1x2x3_min_newt)
print('Pamata funkcijas vertībā pie opt.vert :', pamata_funkcija(a, b, c, d, l, k, *x1x2x3_min_newt))
#print('parc. atv. v. pēc Ņutona alg. pie opt.vert : \n',  hessian_matrica_no_pamata_funkcijas(*x1x2x3_min_newt))


'''
grafiskais attēlojums?
# https://induraj2020.medium.com/implementing-gradient-descent-in-python-d1c6aeb9a448
'''

#==============================================================================
# Scipy minimize function (izmantojot bibliotēku scipy)
# https://www.youtube.com/watch?v=wS5D72wLez8&ab_channel=PhysicsWithNero
#==============================================================================

'''
Lai izmantotu šo iebuvēto bibliotēkas funkciju, vispirms nepieciešams instalēt pašu
scipy bibliotēku, un talāk bus nepieciešams tikai definēt sakuma funkciju un 
ievietot tur izmantotas sakuma punktus. Un tā kā mums jau ir definēta funkcija
paliek tikai izmantot šo optimize.minimize funkciju.
'''

objective_function = lambda x: pamata_funkcija(a, b, c, d, l, k, *x)


x1x2x3_minimize = (0.1, 0.1, 0.1)

# šeit arī būs problēma ar overflow, tapēc ir nepieciešamas robežas katram sakuma punktam
bounds = [(-0.7, 1.2), (-1.6, 1.8), (-10, 0.9)]  
'''
jā mes nedefinēsim metodu, ar kuru stradāsim, tad minimize funkcija, pec noklusējumā izmantos
gradient decent funkciju, kura mums jau ir definēta iepriekš, taču mums vajag kaut
ko citu, tad varam izmantot Powell metodu, kurš arī paradīts video ieraksta.
'''
optimization_result = minimize(objective_function, x1x2x3_minimize, method='Powell', bounds=bounds)
# un, tā ka mums vajag tikai optimizētas vertības un iteraciju skaits, mes izvadīsim tikai tos.
opt_vertibas = optimization_result['x'] 
it = optimization_result['nit']
print ('==============================================================================')
print ('Scipy minimizācijas funkcija')
print ('==============================================================================')
print('Iterāciju skaits =', it)
print('Izvelētās x1,x2,x3 vērtībās : ', x1x2x3_minimize)
print('Optimizētās/minimizētās vērtības : ', opt_vertibas)
print('Pamata funkcijas vertībā pie opt.vert :', pamata_funkcija(a, b, c, d, l, k, *opt_vertibas))


