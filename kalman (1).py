# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 17:05:33 2021

@author: mishagin.k
"""
import numpy as np
from scipy import linalg
import pylab
from scipy.signal import detrend

N = 2018*10
q0 = 10
q1 = 0.1
q2 = 0.01
dt = 1

R = q0/dt
sigma = np.sqrt(R)
v = sigma*np.random.randn(N)
Q = np.array([[q1*dt+q2*dt**3/3, q2*dt**2/2], [q2*dt**2/2, q2*dt]])
L = linalg.cholesky(Q).T
d = 0.01
D = np.array([[d*dt**2/2.],[d*dt]])
print(L)
X = np.array([[0], [0]])
F = np.array([[1, dt], [0, 1]])
phase = np.zeros(N)
for i in range(N):
    a = np.random.randn(2, 1)
    X = F @ X + L @ a + D
    phase[i] = X[0]

phase += v
pylab.figure(1)
pylab.plot(phase)
pylab.xlabel('n')
pylab.ylabel('x(n)')
pylab.title('БФШ + БЧШ + Случайные блуждания частоты')

M = 1
freq = (phase[1:-1:M]-phase[0:-2:M])/dt/M
pylab.figure(2)
pylab.plot(freq)
pylab.xlabel('n')
pylab.ylabel('y(n)')
pylab.title('БФШ + БЧШ + Случайные блуждания частоты')


Xk = np.array([phase[1], (phase[1]-phase[0])/dt])
#F = np.array([[1, dt, dt**2/2], [0, 1, dt], [0, 0, 1]])
P = np.array([[R, 0], [0, 2*R/dt]])
H = np.array([1, 0])
phaseK = np.zeros(N-1)
freqK = np.zeros(N-1)
phaseK[0] = Xk[0]
freqK[0] = Xk[1]
K = np.array([[1], [1]])
for i in range(N-2):
    Xk = F @ Xk
    P = F @ P @ F.T + Q
    
    K = P @ H.T / (H @ P @ H.T + R)
    Xk = Xk + K * (phase[i + 2] - H @ Xk)
    P = (np.eye(2) - np.outer(K, H)) @ P
    phaseK[i+1] = Xk[0]
    freqK[i+1] = Xk[1]

pylab.figure(1)
pylab.plot(phaseK)
pylab.figure(2)
pylab.plot(freqK)