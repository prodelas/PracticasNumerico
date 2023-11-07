#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 17:43:37 2018

@author: pedrogonzalez

Resolvemos un problema de contorno de segundo orden, con condiciones de tipo 
Dirichlet, mediante el MEF unidimensional, usando el E.F. de Lagrange lineal

"""

import numpy as np # Importamos el módulo NumPy con el pseudónimo np
import sympy as sp # Importamos el módulo SymPy con el pseudónimo sp
import matplotlib.pyplot as plt
from scipy.integrate import quad  # 

y,z,t,xL,xR,h_e = sp.symbols('y,z,t,xL,xR,h_e')

def fsp(z):
    return sp.sin(z)

def fnp(z):
    return np.sin(z)

nx= 10;  # número de subintervalos de igual longitud a considerar

a = 0; b = 10; ya = 0; yb = 10;

p1  = ya+(yb-ya)/(b-a)*(z-a)  # este cambio permitiría homogeneizar
p1 # las dos condiciones de contorno de tipo Dirichlet impuestas

x = np.linspace(a,b,nx+1)  # para esta partición concreta

nxx = 1000;  # Esta segunda partición del intervalo se usará como 
xx = np.linspace(a,b,nxx) # plantilla para los gráficos

yf = [fnp(z) for z in x] # valores de la función en los nodos
yyf = fnp(xx)   # valores de la función en la malla de dibujo
yyp1 = np.array([p1.subs({z:x}) for x in xx])


def L(y,z): # usaremos la variable z como variable independiente
    """operador diferencial que define la ec. dif."""
    return -sp.diff(y(z),z,2)
L(y,z)

def yexacta(z):
    """solución exacta del problema tomado como ejemplo"""
    return ya + 1/b*(yb-ya-sp.sin(b))*z+sp.sin(z)

# esta sería pues la solución exacta del problema en este caso


yyexacta = [yexacta(z) for z in xx] 
plt.plot(xx,yyexacta); # gráfica de la solución exacta

# esta nueva función en el segundo miembro de la ec. diferencial
def fbis(z): # permite considerar condiciones de frontera homogéneas
    return (fsp(t)-L(lambda t:p1,t)).subs({t:z})


# definición de las funciones de base del espacio de E.F.

def l1(t):
    """primera func. de base del E.F. de Lagrange P1 en 1D"""
    return 1-t

def l2(t):
    """segunda func. de base del E.F. de Lagrange P1 en 1D"""
    return t

def dl1(t):
    """derivada de la primera func. de base del E.F. de Lagrange P1 en 1D"""
    return -1

def dl2(t):
    """derivada de la segunda func. de base del E.F. de Lagrange P1 en 1D"""
    return 1

def Finv(z,a,b):
    """afinidad entre cada subintervalo de la partición 
        y el intervalo unidad"""
    return (z-a)/(b-a)

def dFinv(z,a,b):
    """ derivada de la afinidad Finv"""
    return 1/(b-a)




def wi(z,x,i):  # en los nodos interiores
    """funciones de base del E.F. de Lagrange P1 unidimensional"""
    if (x[i-1]<=z)*(z<=x[i]):
        valor = l2(Finv(z,x[i-1],x[i]))
    elif (x[i]<=z)*(z<=x[i+1]):  
        valor = l1(Finv(z,x[i],x[i+1]))
    else:
        valor = 0
    return valor

def w0(z,x):  # en el extremo izquierdo del intervalo
    """funcion de base del E.F. de Lagrange P1 unidimensional 
    en el extremo izquierdo"""
    if (x[0]<=z)*(z<=x[1]):
        valor = l1(Finv(z,x[0],x[1]))
    else:
        valor = 0
    return valor

def wn(z,x):  # en el extremo derecho del intervalo
    """funcion de base del E.F. de Lagrange P1 unidimensional 
    en el extremo derecho"""
    # recuérdese que x[-1] y x{-2} indican respectivamente el último y penúltimo nodos
    if (x[-2]<=z)*(z<=x[-1]):  
        valor = l2(Finv(z,x[-2],x[-1])) 
    else:
        valor = 0
    return valor

def dwi(z,x,i):  # en los nodos interiores
    """derivadas de las funcs. de base del E.F. de Lagrange P1 1D"""
    if (x[i-1]<=z)*(z<=x[i]):
        valor = dFinv(z,x[i-1],x[i])  #  dl2 = +1
    elif (x[i]<=z)*(z<=x[i+1]):  
        valor = -dFinv(z,x[i],x[i+1]) #  dl1 = -1
    else:
        valor = 0
    return valor

def dw0(z,x):  # en el extremo izquierdo del intervalo
    """derivada de la func. de base del E.F. de Lagrange P1 1D
    en el extremo izquierdo"""
    if (x[0]<=z)*(z<=x[1]):
        valor = -dFinv(z,x[0],x[1])  #  dl1 = -1
    else:
        valor = 0
    return valor

def dwn(z,x):  # en el extremo derecho del intervalo
    """derivada de la func. de base del E.F. de Lagrange P1 1D 
    en el extremo derecho"""
    if (x[-2]<=z)*(z<=x[-1]):         # x[-2]=x[n-1], x[-1]=x[n]
        valor = dFinv(z,x[-2],x[-1])  #  dl2 = +1
    else:
        valor = 0
    return valor



# plt.plot(xx,yf);


# de rigidez local en un intervalo genérico [xL,xR]
Agen = sp.Matrix([[1/h_e,-1/h_e],[-1/h_e,1/h_e]])

print(Agen)


# ensamblaje de la matriz global
A = np.zeros((nx-1,nx-1),dtype=float)

# primero calcularemos los elementos de la diagonal principal
for i in range(1,nx): 
    A[i-1,i-1] = Agen[1,1].subs({h_e:x[i]-x[i-1]})+Agen[0,0].subs({h_e:x[i+1]-x[i]})
    
for i in range(1,nx-1): # y ahora la diagonal inferior adyacente
    A[i-1,i] = Agen[0,1].subs({h_e:x[i+1]-x[i]})
    A[i,i-1] = A[i-1,i] # aprovechando la simetría de la matriz
    
print(A) # esta sería la matriz de rigidez global en este caso

B = np.zeros(nx-1); # vamos ahora con el vector de términos independientes
for i in range(1,nx):  # Atención: ¡cuidado con los índices en Python!
    intizda = quad(lambda z:fbis(z)*l2(Finv(z,x[i-1],x[i])),x[i-1],x[i])
    intdcha = quad(lambda z:fbis(z)*l1(Finv(z,x[i],x[i+1])),x[i],x[i+1])
    B[i-1] = intizda[0] + intdcha[0]
print(B)

C = np.linalg.solve(A,B)  # resolución del sistema lineal

print(C)

yy = np.array([sum([wi(z,x,i)*C[i-1] for i in range(1,nx)]) for z in xx])
# yy = yy +    ya*np.array([w0(z,x) for z in xx]) 
# yy = yy +    yb*np.array([wn(z,x) for z in xx])
# plt.plot(xx,yy); # nótese que habrá que sumar p1(z) para que se 
plt.plot(xx,yy + yyp1); # satisfagan las condiciones de contorno
yyexacta = [yexacta(z) for z in xx] 
plt.plot(xx,yyexacta);  # y aproximar la solución exacta del pbma.
