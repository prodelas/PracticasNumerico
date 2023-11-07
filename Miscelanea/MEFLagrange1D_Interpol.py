#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 17:43:37 2018

@author: pedrogonzalez
"""

import numpy as np # Importamos el módulo NumPy con el pseudónimo np
import sympy as sp # Importamos el módulo SymPy con el pseudónimo sp
import matplotlib.pyplot as plt

def f(z):
    return np.sin(z)

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

nx= 10;  # número de subintervalos de igual longitud a considerar
# definición de las funciones de base del espacio de E.F.
a = 0; b = 10;
x = np.linspace(a,b,nx+1)  # para esta partición concreta

nxx = 100; # malla más fina para dibujar la función
xx =np.linspace(a,b,nxx+1) 

y = [f(z) for z in x] # valores de la función en los nodos
yf = f(xx)   # valores de la función en la malla de dibujo

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

nxx = 1000;  # Esta segunda partición del intervalo 
xx = np.linspace(a,b,nxx)

fig,ax = plt.subplots(figsize=(8,8))

for i in range(1,nx):
    yy = [wi(z,x,i) for z in xx]
    plt.plot(xx,yy);
yy = [w0(z,x) for z in xx]
plt.plot(xx,yy);
yy = [wn(z,x) for z in xx]
plt.plot(xx,yy);

ax.set_xlabel('$x$',fontsize=18)
ax.set_ylabel('$y$',fontsize=18)
ax.set_title('Funciones de base del E.F. de Lagrange P1 en 1D');

fig,ax = plt.subplots(figsize=(8,8))
i=nx//2;  # nodo intermedio
yy = [wi(z,x,i-1) for z in xx]
plt.plot(xx,yy,label='$w_{i-1}(x)$');
yy = [wi(z,x,i) for z in xx]
plt.plot(xx,yy,label='$w_i(x)$');
yy = [wi(z,x,i+1) for z in xx]
plt.plot(xx,yy,label='$w_{i+1}(x)$');

ax.set_xticks([x[i-2],x[i-1], x[i], x[i+1], x[i+2]])
ax.set_xticklabels(['$x_{i-2}$','$x_{i-1}$','$x_{i}$','$x_{i+1}$','$x_{i+2}$'], fontsize=18)
ax.spines['bottom'].set_position(('data',0)) 
ax.spines['left'].set_position(('data',0))

# fig,ax = plt.subplots(figsize=(8,8))
ym = -10; yi = 10; yp =2; # cambie estos valores para ver el efecto final
yy = [ym*wi(z,x,i-1)+yi*wi(z,x,i)+yp*wi(z,x,i+1) for z in xx]
plt.plot(xx,yy,label='$y_{i-1}w_{i-1}(x)+y_{i}w_{i}(x)+y_{i+1}w_{i+1}(x)$');
ax.set_xlabel('$x$',fontsize=18)
ax.set_ylabel('$y$',fontsize=18)
ax.legend(loc=4)
ax.set_title('Combinación de varias funciones de base del E.F. de Lagrange P1 en 1D');

fig,ax = plt.subplots(figsize=(8,8))
yy = np.array([sum([wi(z,x,i)*y[i] for i in range(1,nx)]) for z in xx])
yy = yy + np.array([wn(z,x)*y[-1] for z in xx])
plt.plot(xx,yy);  # ¡nótese lo que ocurre en el extremo derecho!
yf = [f(z) for z in xx]
plt.plot(xx,yf);