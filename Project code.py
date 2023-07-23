# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 17:50:18 2023

@author: fayew
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from sympy import *
import scipy.integrate as spi 

#Define values
w=800/31 #river width
slope=-0.001 #river bed slope
g=9.81 #acceleration due to gravity constant
Q_0=100 #discharge
C_m=0.03 #Manning coefficient
P_w=2 #weir height
C_f = (2/3)**(3/2) #discharge coefficient
s_0=0 #where weir plot begins
sigma = 200 # scale of weir
s_x = 1100 # location of weir's crest

#Find h_0 by solving Q_0 = ... replacing u with constants via Manning relation
f = lambda h_0: h_0*w * (h_0*w/(2*h_0 + w))**(2/3) * ((-slope)**(1/2) / C_m) - Q_0
h_0_ans = fsolve(f, [0, 100])
h_0 = h_0_ans[0]

h_1 = (Q_0/(g**(1/2)*w))**(2/3)
#Original estimate for height just before weir, excluding u term, h_e
hepsilon = lambda h: h - (3*Q_0**(2/3))/(2*g**(1/3)*w**(2/3)) - P_w
h_e_ans = fsolve(hepsilon, [-10000,10000])
h_e = h_e_ans[0]
#O.E.
h_e_0 = (Q_0/(g**(1/2)*C_f*w))**(2/3) + P_w

#Head of water over the weir
H = h_e - P_w

#Updated estimate for height just before weir including u term, h_e2
hepsilon2 = lambda h: h - (3*Q_0**(2/3))/(2*g**(1/3)*w**(2/3)) + (Q_0**2/(2*g*h**2*w**2)) - P_w
h_e2_ans = fsolve(hepsilon2, [-10000,10000])
h_e2 = h_e2_ans[0]

#Updated value for head of water over weir
H_2 = h_e2 - P_w

#Error of h_e estimate relative to h_e2 estimate
error = abs((h_e2-h_e)/h_e)
error2 = abs((H_2-H)/H)

#Length from s_1 (where upstream and downstream flow meet) to weir
length_s = abs((h_e-h_0)/slope)
#Location of s_1
s_1 = s_x - length_s
#y-intercept of the river bed, decided by taking s_x as the point at which the river slope =0
coef = -slope*(s_x)

#Height function relative to x-axis
def hb(s):
    if s<s_1:
        return h_0 + -(-slope*s)+coef
    else:
        return P_w + H
s = np.linspace(s_0,s_x,10000)

#Points s_a and s_b are the points at which the river bed and weir meet,
#found by setting the weir equation equal to the river bed equation
s_meet = lambda s: -(s-s_x)**2/sigma + P_w - (coef + slope*s)
s_meet_ans = fsolve(s_meet, [0,100000])
s_a = s_meet_ans[0]
s_b = s_meet_ans[1]
#Length of weir over s
s_dist = s_b-s_a

#Full eq for river bed b(s), using the points at which the weir and bed meet, s_a and s_b
def b(x):
 if(x < s_a): 
     return coef+slope*x
 if(x > s_b): 
     return coef+slope*x
 else: 
     return (-(x-s_x)**2/sigma)+P_w
x=np.linspace(s_0, s_b+50, 10000)

#continuation of line of intial slope for reference
h_0_cont = h_0 + -(-slope*s)+coef

#Estimate for A*, h*, hydraulic radius and dsb at the point A=A*, s=s_w using shallow water theory
A_star = (Q_0**2*w/g)**(1/3)
h_star = A_star/w
R = A_star/((2*A_star/w) + w)
dsb = -((C_m**2)*(Q_0**2))/((A_star**2)*(R**(4/3)))

#Formula for s_w, point at which criticality holds, using estimate for dsb given at s=s_w via T(A*)=0
s_w1 = lambda s: -2*(s-s_x)/sigma - dsb
s_w_ans = fsolve(s_w1, [0,10000])
s_w = s_w_ans[0]

Ns=50000

#Height of the weir at s=s_w
height_weir = -((s_w-s_x)**2/sigma)+P_w

##############################################################################
#Plot of river profile
##########################################################################

plt.figure(1)
plt.plot(s, list(map(hb, s)), 'r-')
plt.plot(x, list(map(b, x)), 'k-')
plt.plot(s, h_0_cont, 'r:')
plt.plot([s_x,s_x], [0,P_w],  color='orange')
plt.plot([s_w,s_w], [height_weir,height_weir],  color='green', marker = 'x', markersize=7)
plt.plot([s_w,s_w], [h_star+P_w,h_star+P_w],  color='purple', marker = 'x', markersize=7)
plt.legend(['h(s)+b(s)','b(s)', 'h(s)+b(s) without weir','Initial weir',  'b(s = $\mathregular{s_w}$)', 'h = $\mathregular{h^*}$'])
plt.xlabel("s (m)")
plt.ylabel("h(s) (m)")
plt.show()

length_s_2 = abs((H_2+P_w-h_0)/slope)
s_1_2 = s_x - length_s_2 #Point at which water flowing from river and water held back by weir meet
coef_2 = -slope*(s_x)

def h2(s):
    if s<s_1_2:
        return h_0 + -(-slope*s)+coef_2
    else:
        return P_w + H_2
    
h_0_2_cont = h_0 + -(-slope*s)+coef_2

#####################################################################################
#Plot of river profile for exact estimate of height immediately before the weir, h_e2
#####################################################################################

plt.figure(2)
plt.plot(s, list(map(hb, s)), 'r-')
plt.plot(x, list(map(b, x)), 'k-')
plt.plot(s, list(map(h2, s)), color="springgreen")
plt.plot(s, h_0_2_cont, linestyle='dotted',  color="springgreen")
plt.plot([s_x,s_x], [0,P_w],  color='orange')
plt.plot([s_w,s_w], [height_weir,height_weir],  color='green', marker = 'x', markersize=7)
plt.plot([s_w,s_w], [h_star+P_w,h_star+P_w], color='purple', marker = 'x', markersize=7)
plt.legend(['Original h(s)+b(s)','b(s)','New h(s)+b(s)', 'h(s)+b(s) without weir', 'Initial weir', 's = $\mathregular{s_w}$', 'h = $\mathregular{h^*}$'])
plt.xlabel("s (m)")
plt.ylabel("h(s) (m)")
plt.show()

##############################################################################
#Plot of zoomed in weir section of river profile
##########################################################################

plt.figure(3)
plt.plot(s, h_0_cont, 'r:')
plt.plot(x, list(map(b, x)), 'k-')
plt.plot([s_w,s_w], [height_weir,height_weir], color='green', marker = 'x', markersize=10)
plt.plot([s_x,s_x], [0,P_w],  color='orange')
plt.plot([s_w,s_w], [h_star+P_w,h_star+P_w], color='purple', marker = 'x', markersize=10)
plt.legend(['h(s)+b(s) without weir','b(s)', 's = $\mathregular{s_w}$', 'Initial weir', 'h = $\mathregular{h^*}$'])
plt.xlabel("s (m)")
plt.ylabel("h(s) (m)")
plt.xlim(s_a-50, s_b+50)
plt.ylim(-0.25, 2.5)
plt.show()


##############################################################################
#Plot of river profile via backwards integration
#############################################################################

#Approximation of dA/ds via L'Hopital's rule
A = Symbol('A')
y_1 = -g*A*(dsb + ((C_m**2*Q_0**2*(2*A/w + w)**(4/3))/A**(10/3))) #T(A)
df_A_t = y_1.diff(A) #T'(A)
y_2 = (g*A)/w - Q_0**2/A**2 #B(A)
df_A_b = y_2.diff(A) #B'(A)

#Evaluating this at A=A*, we have
df_A_start = df_A_t.subs(A, A_star) #T'(A*)
df_A_starb = df_A_b.subs(A, A_star) #B'(A*)
df_split = df_A_start/df_A_starb #T'(A*)/B'(A*)

#Number of points along s we shall find the cross sectional area of
Ns = 50000
#Section of s each estimate of A will cover
ds = (s_w-s_0)/Ns

#Have initial estimate A* and using L'Hopital's rule the second estimate, A_49999 is given by
A_49999 = A_star + ds * abs(df_split)

plt.figure(4)
#T(A)/B(A) for river bed before s_a and weir after s_a
def f(A, s):
    if s < s_a:
        return (-g*A*(slope+ C_m**2*Q_0**2*(2*A/w + w)**(4/3)/A**(10/3))) / (g*A/w - Q_0**2/A**2)
    else:
        return (-g*A*(-(2*(s-s_x)/sigma)+ (C_m**2*Q_0**2*(2*A/w + w)**(4/3)/A**(10/3)))) / (g*A/w - Q_0**2/A**2)

#Points along s that we will find cross sectional area of    
s_values = np.linspace(s_0, s_w, Ns)
A = 0.0*s_values
bb = 0.0*s_values
A[Ns-1] = A_star
A[Ns-2] = A_49999
bb[Ns-1]= b(s_w)
bb[Ns-2]= b(s_w - ds)
#Loop to carry out iterative formula to find A and to find b(s) along s (to check)
for i in range(Ns-3, 0, -1):
    A[i] = A[i+1] - (ds * f(A[i+1], s_values[i+1]))
    bb[i] = b(s_values[i+1])

plt.plot(x, list(map(b, x)), 'k')#to check they align
plt.plot(s, list(map(hb, s)), 'r-')
plt.plot(s, h_0_cont, 'r:')
plt.plot(s_values, (A/w)+bb, color='steelblue', ls='--')
plt.plot([s_x,s_x], [0,P_w], color='orange')
plt.plot([s_w,s_w], [height_weir,height_weir], color='green', marker = 'x', markersize=7)
plt.plot([s_w,s_w], [h_star+P_w,h_star+P_w], color='purple', marker = 'x', markersize=7)
plt.xlabel('s (m)')
plt.ylabel('h(s) (m)')
plt.legend(['Original b(s)', 'Original h(s)+b(s)', 'h(s)+b(s) without weir', 'Calculated h(s)+b(s)', 'Initial weir', 's = $\mathregular{s_w}$', 'h = $\mathregular{h^*}$'])
plt.xlim(s_0,s_w+50)
plt.show()


##############################################################################
#Plot of river profile via forwards integration
#############################################################################

plt.figure(5)

#Using same Ns, ds and s_values as seen in the backwards integration method
A2 = 0.0*s_values
bb2 = 0.0*s_values
#Initial estimate for A and b(s) at s=Y
A2[0] = h_0*w
bb2[0]= b(s_0)

#Loop to find cross sectional area and b(s) values along s
for ii in range(1, Ns-1, 1):
    A2[ii] = A2[ii-1] + ds*f(A2[ii-1], s_values[ii-1])
    bb2[ii] = b(s_values[ii-1] + ds)
plt.plot(s_values, list(map(b, s_values)), 'k-') #to check they align
plt.plot(s_values, list(map(hb, s_values)), 'r-')
plt.plot(s, h_0_cont, 'r:')
plt.plot(s_values, (A2/w)+bb2, color='steelblue', ls='--')
plt.plot(s_values, bb2, 'y--')
plt.plot([s_x,s_x], [0,P_w], color='orange')
plt.plot([s_w,s_w], [height_weir,height_weir], color='green', marker = 'x', markersize=7)
plt.plot([s_w,s_w], [h_star+P_w,h_star+P_w], color='purple', marker = 'x', markersize=7)
plt.xlabel('s (m)')
plt.ylabel('h(s) (m)')
plt.legend(['h(s)+b(s)','h(s)', 'b(s)'])
plt.legend(['Original b(s)', 'Original h(s)+b(s)', 'h(s)+b(s) without weir', 'Calculated b(s)', 'Calculated h(s)+b(s)', 'Initial weir', 's = $\mathregular{s_w}$', 'h = $\mathregular{h^*}$'])
plt.xlim(s_0,s_w+50)
plt.ylim(0,6)
plt.show()

########################################################################################################
#Plot of velocity and sqrt(gh) along s, using the value of h along s determined by backwards integration
########################################################################################################

plt.figure(6)
plt.plot(s_values, list(map(b, s_values)), color="black", alpha=0.5)
plt.plot(s_values, list(map(hb, s_values)), color="red", alpha=0.5)
plt.plot(s_values, (A/w)+bb, '--', color="red", alpha=0.5)
plt.plot(s_values, ((A/w)*g)**(1/2), color='teal')
plt.plot(s_values, Q_0/(A), color='darkorange')
plt.plot([s_w,s_w], [h_star+P_w,h_star+P_w], color='purple', marker = 'x', markersize=7)
plt.legend(['Original b(s)', 'Original h(s)+b(s)', 'New h(s)+b(s)', 'u=$\mathregular{\sqrt{gh}}$', 'u=Q/A', 'h = $\mathregular{h^*}$'])
plt.xlim(s_0,s_w+50)
plt.xlabel('s (m)')
plt.ylabel('h(s) (m)')

########################################################################################################
#Plot of mesh over weir up to s_w
########################################################################################################

plt.figure(7)
X3 = np.linspace(s_a, s_w, 20)
result = [spi.quad(lambda x: b(x), c, s_w)[0] for c in X3]
plt.axvline(x=s_x, color='k', ls='--')
plt.plot(X3, result, 'm')
plt.xlabel("s (m)")
plt.ylabel("Integral of b(s)")

########################################################################################################
#Plot of mesh over the whole of s, up to s_w
########################################################################################################

plt.figure(8)
X4 = np.linspace(s_0, s_a, 200)
result2 = [spi.quad(lambda x: b(x), c, s_w)[0] for c in X4]
plt.plot(X4, result2, 'g')
plt.plot(X3, result, 'm')
plt.axvline(x=s_x, color='k', ls='--')
plt.xlabel("s (m)")
plt.ylabel("Integral of b(s)")
plt.legend(['b(s) before weir', 'b(s) at weir', 's=$\mathregular{s_x}$'])

########################################################################################################
#Flood storage volume calculations from start of river profile up tp crest of the weir, s_x
########################################################################################################

#FSV calculated from the original h_e approximation
FSV_1 = w*(h_e-h_0)*(s_x-s_1)/2
#FSV calculated from the refined h_e2 estimation
FSV_1_e = w*(h_e2-h_0)*(s_x-s_1)/2
#Volume underneath the backwards integration line
Vol_beta = sum(A)*ds

#Backwards integration method to find the volume underneath the section s_x to s_w, 
#as this will not be included in the FSV calculation
Ns2=5000 #Smaller value as we are looking at a smaller section of s
ds2 = (s_w-s_x)/Ns2
A_4999 = A_star + ds * abs(df_split)

s_values2 = np.linspace(s_x, s_w, Ns2)

A3 = 0.0*s_values2
bb3 = 0.0*s_values2
A3[Ns2-1] = A_star
A3[Ns2-2] = A_4999
bb3[Ns2-1]= b(s_w)
bb3[Ns2-2]= b(s_w - ds2)
for i in range(Ns2-2, 0, -1):
    A3[i] = A3[i+1] - (ds2 * f(A3[i+1], s_values2[i+1]))
    bb3[i] = b(s_values2[i+1]) 
Vol_gamma = sum(A3)*ds2

Vol_1 = Vol_beta - Vol_gamma
Vol_2 = h_0*(s_w-s_0)*w
FSV_2 = Vol_1 - Vol_2
FSV_error = abs(FSV_1-FSV_2)/FSV_2

###########################################################################
#Plotting river heights h_0, h_e, h_e2 for different discharge rates
##########################################################################

from scipy.optimize import root
#Define values for Q we want to find FSV at
Q_values = np.arange(50,500,50)
#Keep the same weir height as previously seen
P_w=2

#Finding h_0 values for corresponding discharge value
def func(h_0, Q):
    return h_0*w * (h_0*w/(2*h_0 + w))**(2/3) * ((-slope)**(1/2) / C_m) - Q
h_0_values = []
for Q in Q_values:
    sol = root(func, 1, args=(Q,))
    h_0_values.append(list({sol.x[0]}))
    
#Finding h_e values for corresponding discharge value
def func(h_e, Q):
    return h_e - ((3*Q**(2/3))/(2*g**(1/3)*w**(2/3)) + P_w)
h_e_values = []
for Q in Q_values:
    soly = root(func, 1, args=(Q,))
    h_e_values.append(list({soly.x[0]}))

#Finding h_e2 values for corresponding discharge value
def func(h_e2, Q):
    return h_e2 - (3*Q**(2/3))/(2*g**(1/3)*w**(2/3)) + (Q**2/(2*g*h_e2**2*w**2)) - P_w
h_e2_values = []
for Q in Q_values:
    solx = root(func, 3, args=(Q,))
    h_e2_values.append(list({solx.x[0]}))

###########################################################################
#Plotting flood storage volume for different discharge rates
##########################################################################

#To find FSV_1 = w*(h_e-h_0)**2/(2*-slope) we require h_0 and h_e values 
#at each discharge rate, found in section above
#Meanwhile, slope and w remain the same

#To find h_e - h_0 = h_diff
h_0_values = np.array(h_0_values)
h_diff_values = []
h_0_neg = list(-h_0_values)
for x, y in zip(h_e_values, h_0_neg):
    h_diff_values.append(x+y)
FSV_Q_values = ((np.array(h_diff_values)**2)/(2*-slope))*w

#For h_e2
h_0_values = np.array(h_0_values)
h_2_diff_values = []
for x, y in zip(h_e2_values, h_0_neg):
    h_2_diff_values.append(x+y)
FSV_Q_he2_values = ((np.array(h_2_diff_values)**2)/(2*-slope))*w

plt.figure(9)
FSV_2_Q = [20479.06438, 22577.97906, 24273.14242, 25865.54181, 27194.82248, 28282.71072, 29166.09209, 29950.03348, 30778.02576]
plt.plot(Q_values, FSV_Q_values, color = 'r', marker='^')
plt.plot(Q_values, FSV_Q_he2_values, color = 'springgreen', marker='s')
plt.plot(Q_values, FSV_2_Q, color = 'steelblue', marker='H')
plt.xlabel("Discharge ($\mathregular{m^3/s}$)")
plt.ylabel("Flood Storage Volume ($\mathregular{m^3}$)")
plt.legend(["Flood Storage Volume from approximation, $\mathregular{FSV_1}$ with small u approximation $\mathregular{h_\epsilon}$", "Flood Storage Volume from approximation, $\mathregular{FSV_1}$ with exact $\mathregular{h_{\epsilon}}$", "Flood Storage Volume from integration, $\mathregular{FSV_2}$"], loc='lower right')
plt.ylim(0, 35000)

##############################################################################
#Plotting flood storage volume for different weir heights up to s_x
##########################################################################

#Want FSV_1 = w*(h_e-h_0)**2/(2*-slope)
#slope, w and h_0 do not depend on P_w, so we must find h_e and h_e2 only

plt.figure(10)
#Define constant discharge
Q_0=100
#Values of P_w we shall use
P_w_values = np.arange(0.5,6,.5)

#Values of h_e2 for different weir heights, take equal to h_0 at P_w=0 i.e. normal stream
def funcx(h_e2, P_w):
    return h_e2 - (3*Q_0**(2/3))/(2*g**(1/3)*w**(2/3)) + (Q_0**2/(2*g*h_e2**2*w**2)) - P_w
h_e2_Pw_values = []
for P_w in P_w_values:
    solx = root(funcx, 3, args=(P_w,))
    h_e2_Pw_values.append(list({solx.x[0]}))

#Find values of h_e for different P_w values
def funcx(h_e_Pw, P_w):
    return h_e_Pw - (3*Q_0**(2/3))/(2*g**(1/3)*w**(2/3)) - P_w
h_e_Pw_values = []
for P_w in P_w_values:
    solx = root(funcx, 3, args=(P_w,))
    h_e_Pw_values.append(list({solx.x[0]}))

#Equation for FSV of approximation using h_e    
FSV_Pw_values = (h_e_Pw_values - h_0)**2*w/(2*-slope)
#Equation for FSV of approximation using h_e2
FSV_Pw_he2_values = (h_e2_Pw_values - h_0)**2*w/(2*-slope)
#Values for FSV of backwards integration for different P_w values
FSV_2_Pw = [1706.293784, 6523.688909, 13484.32036, 22577.97906, 33941.84605, 47422.63388, 63321.72805, 81656.21226, 101908.1181, 124529.9528, 149764.1619]
plt.plot(P_w_values, FSV_Pw_values, 'r', marker='^')
plt.plot(P_w_values, FSV_Pw_he2_values, color='springgreen', marker='s')
plt.plot(P_w_values, FSV_2_Pw, color='steelblue', marker='H')
plt.legend(['Flood Storage Volume from approximation, $\mathregular{FSV_1}$ with $\mathregular{h_e}$ approximation', 'Flood Storage Volume from approximation, $\mathregular{FSV_1}$ with exact $\mathregular{h_e}$', 'Flood Storage Volume from integration, $\mathregular{FSV_2}$'])
plt.xlabel("Weir Height, $\mathregular{P_w}$ (m)")
plt.ylabel("Flood Storage Volume, $\mathregular{V_s}$ ($\mathregular{m^3}$)")

##############################################################################
#Plotting flood storage volume for different slopes up to weir crest s_x
##########################################################################

#Want FSV_1 = w*(h_e-h_0)**2/(2*-slope)
#w, h_e and h_e2 do not depend on slope, but h_0 does

#Define constants and slope values
Q_0=100
P_w=2
slope_values = np.linspace(-0.001,-.009, 9)

#Finding h_0 for each slope value
def func(h_0, slope_s):
    return h_0*w * (h_0*w/(2*h_0 + w))**(2/3) * ((-slope_s)**(1/2) / C_m) - Q_0
h_0_values_slope = []
for slope_s in slope_values:
        sol = root(func, 1, args=(slope_s,))
        h_0_values_slope.append(list({sol.x[0]}))
    
#Find the difference between h_e and h_0 squared for each value of h_0
height_diff_slope = (h_e-h_0_values_slope)**2 
#Put into FSV_1 equation
slope_values_denom = list(-2*np.array(slope_values))
FSV_values_slope = []
for x, y in zip(height_diff_slope, slope_values_denom ):
    FSV_values_slope.append((x/y)*w)
    
#Find the difference between h_e and h_0 squared for each value of h_0
height_diff_he2_slope = (h_e2-h_0_values_slope)**2 
#Put into FSV_1 equation)
FSV_values_he2_slope = []
for x, y in zip(height_diff_he2_slope, slope_values_denom ):
    FSV_values_he2_slope.append((x/y)*w)
    
plt.figure(11)
FSV_2_slope = [41906.38517, 28047.84924, 21085.51063, 16931.3361, 14157.6442, 12106.3664, 10613.25381, 9445.317199, 8503.879762]
plt.plot(slope_values, FSV_values_slope, color = 'r', marker='^')
plt.plot(slope_values, FSV_values_he2_slope, color = 'springgreen', marker='s')
plt.plot(slope_values, FSV_2_slope, color = 'steelblue', marker='H')
plt.xlabel("Slope (tan \u03B1)")
plt.ylabel('Flood Storage Volume ($\mathregular{m^3}$)')
plt.legend(["Flood Storage Volume from approximation, $\mathregular{FSV_1}$, with $\mathregular{h_\epsilon}$ approximation", "Flood Storage Volume from approximation, $\mathregular{FSV_1}$ with exact $\mathregular{h_\epsilon}$", "Flood Storage Volume from integration, $\mathregular{FSV_2}$"])
