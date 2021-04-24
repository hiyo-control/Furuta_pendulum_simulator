# ***************************************************************

# Furuta pendululm simulator

# ***************************************************************

import math
import yaml
import numpy
import control
import matplotlib.pyplot as plt

# ===============================================================

# Parameters definition

# ===============================================================
with open('parameter_list.yml', 'r') as yml:
    parameter = yaml.safe_load(yml)

# arm parameters
ma = parameter['arm']['mass']
lap = parameter['arm']['distance']
da = parameter['arm']['viscosity']
Ja = parameter['arm']['inertia_moment']

# pendulum parameters
mp = parameter['pendulum']['mass']
lpg = parameter['pendulum']['distance']
dp = parameter['pendulum']['viscosity']
lp = parameter['pendulum']['length']
Jp = mp * lp**2 / 3  # inertia moment

# motor parameters
kv = parameter['motor']['bmf_const']
ktau = parameter['motor']['torque_const']
R = parameter['motor']['resistance']
kg = parameter['motor']['reduction_ratio']

g = 9.81  # gravity acceleration

# ===============================================================

# Controller design
# linearized state equation : dot_x=Ax+Bv
# x = (theta_a, theta_p, dot_theta_a, dot_theta_p)

# ===============================================================

# determinant of inertia matrix
detM = Jp * (Ja + mp * lap**2) - (mp * lap * lpg)**2
# total negative FB of arm angle velocity
dam = da + ktau * kv * kg**2 / R

# ---------------------------------------------------------------
# A matrix (continuous)
# ---------------------------------------------------------------
a32 = -(mp**2) * lap * (lpg**2) * g / detM
a33 = -dam * Jp / detM
a34 = dp * mp * lap * lpg / detM

a42 = (Ja + mp * lap**2) * mp * lpg * g / detM
a43 = dam * mp * lap * lpg / detM
a44 = -dp * (Ja + mp * lap**2) / detM

A = [
    [0, 0, 1, 0],
    [0, 0, 0, 1],
    [0, a32, a33, a34],
    [0, a42, a43, a44]
]
A = numpy.array(A)  # create A matrix

# ----------------------------------------------------------------
# B matrix (continuous)
# ----------------------------------------------------------------

B = [
    [0],
    [0],
    [(kg * ktau / R) * Jp / detM],
    [-(kg * ktau / R) * (mp * lap * lpg) / detM]
]
B = numpy.array(B)  # create B matrix

# ---------------------------------------------------------------
# Ad matrix (discrete)
# ---------------------------------------------------------------

T = 0.01  # sampling interval

Ad = numpy.zeros((len(A), len(A)))
termA = numpy.eye(len(A))
for i in range(10):
    Ad = Ad + termA / math.factorial(i)
    termA = numpy.dot(termA, (A * T))

# ---------------------------------------------------------------
# Bd matrix (discrete)
# ---------------------------------------------------------------

Bd_temp = numpy.zeros((len(A), len(A)))
termB = numpy.eye(len(A))
for i in range(10):
    Bd_temp = Bd_temp + termB / math.factorial(i)
    termB = numpy.dot(termB, (A * T))
Bd_temp = numpy.dot(Bd_temp, T)
Bd = numpy.dot(Bd_temp, B)

# ---------------------------------------------------------------
# calculate optimal linear state FB gain vector
# ---------------------------------------------------------------
with open('weighting.yml', 'r') as yml:
    weighting = yaml.safe_load(yml)

# weighting matrix Q for state vector
Q = numpy.array([
    [weighting['state']['arm_angle'], 0, 0, 0],
    [0, weighting['state']['pendulum_angle'], 0, 0],
    [0, 0, weighting['state']['arm_angle_velocity'], 0],
    [0, 0, 0, weighting['state']['pendulum_angle_velocity']]
])

# weight for control input
R = weighting['input']

# discretization
Qd = Q * T
Rd = R * T

# solve Riccati equation
P, L, F = control.dare(Ad, Bd, Qd, Rd)
# P:solution of Riccati equation
# L:eigen value of closed loop system
# F:linear state Fb gain vector

# ===============================================================

# Furuta pendulum simulator

# state vector
# x = (x1,x2,x3,x4)
#   = (theta_a, phi_p, dot_theta_a, dot_phi_p)

# Non-lienar equation of motion
# M(x2)ddot_[x3 \\ x4] = - D[x3 \\ x4] + Bv + h(x2,x3,x4)

# Non-linear state equation
# dot_x = f(x) + gx(x)V

# ===============================================================

# define drift term f(x)


def f(x1, x2, x3, x4):
    # elements of inertia matrix M(x2)
    m11 = Ja + mp * lap**2 + Jp * math.sin(x2)**2
    m12 = mp * lap * lpg * math.cos(x2)
    m22 = Jp

    # determinant of inertia matrix M(x2)
    detM_x = (Ja + mp * lap**2 + Jp * math.sin(x2)**2) * Jp \
        - (mp * lap * lpg * math.cos(x2))**2

    h1 = Jp * math.sin(2 * x2) * x3 * x4 \
        + mp * lap * lpg * math.sin(x2) * x4**2
    h2 = Jp * math.sin(x2) * math.cos(x2) * x3**2 \
        + mp * lpg * g * math.sin(x2)

    # elements of vicsocity matrix D
    d11 = da + ktau * kv * kg**2 / R
    d22 = dp

    # elements of drift term fx(x)
    f1 = x3
    f2 = x4
    f3 = (- m22 * d11 * x3 + m12 * d22 * x4) / detM_x \
        + (m22 * h1 - m12 * h2) / detM_x
    f4 = (m12 * d11 * x3 - m11 * d22 * x4) / detM_x \
        + (- m12 * h1 + m11 * h2) / detM_x

    return numpy.array([f1, f2, f3, f4])

# define input distribute vector gx(x)


def gx(x1, x2, x3, x4):
    # elements of inertia matrix M(x)
    m12 = mp * lap * lpg * math.cos(x2)
    m22 = Jp

    # determinant of inertia matrix M(x)
    detM_x = (Ja + mp * lap**2 + Jp * math.sin(x2)**2) * Jp \
        - (mp * lap * lpg * math.cos(x2))**2

    # elements input destribute vector
    g1 = 0
    g2 = 0
    g3 = (kg * ktau / (R * detM_x)) * m22
    g4 = - (kg * ktau / (R * detM_x)) * m12

    return numpy.array([g1, g2, g3, g4])


# ---------------------------------------------------------------
# Numerial calculation by 4the order Runge-Kutta method
# ---------------------------------------------------------------

# simulation time [sec]
T_end = 5

# define initial state
with open('initial_state.yml', 'r') as yml:
    initial = yaml.safe_load(yml)

x = numpy.array([
    initial['arm_angle'] * math.pi / 180,
    initial['pendulum_angle'] * math.pi / 180,
    initial['arm_angle_velocity'] * math.pi / 180,
    initial['pendulum_angle_velocity'] * math.pi / 180
])

# define array for graph & input initial state
time = [0]
x1 = [x[0] * 180 / math.pi]
x2 = [x[1] * 180 / math.pi]
x3 = [x[2] * 180 / math.pi]
x4 = [x[3] * 180 / math.pi]
V = [0]

# numerical calculation for Non-linear differential equation
for i in range(round(T_end / T)):
    Vin = numpy.dot(-F, x)
    if Vin > 6:
        Vin = 6
    if Vin < -6:
        Vin = -6

    k1 = f(x[0], x[1], x[2], x[3]) \
        + gx(x[0], x[1], x[2], x[3]) * Vin
    a = x + T * k1 / 2
    k2 = f(a[0], a[1], a[2], a[3]) \
        + gx(a[0], a[1], a[2], a[3]) * Vin
    a = x + T * k2 / 2
    k3 = f(a[0], a[1], a[2], a[3]) \
        + gx(a[0], a[1], a[2], a[3]) * Vin
    a = x + T * k3
    k4 = f(a[0], a[1], a[2], a[3]) \
        + gx(a[0], a[1], a[2], a[3]) * Vin

    x = x + T * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    x1.append(x[0] * 180 / math.pi)
    x2.append(x[1] * 180 / math.pi)
    x3.append(x[2] * 180 / math.pi)
    x4.append(x[3] * 180 / math.pi)
    time.append(T * i)
    V.append(Vin)

# ---------------------------------------------------------------
# Graph
# ---------------------------------------------------------------

fig = plt.figure(figsize=(9, 9))

ax1 = fig.add_subplot(5, 1, 1)
ax2 = fig.add_subplot(5, 1, 2)
ax3 = fig.add_subplot(5, 1, 3)
ax4 = fig.add_subplot(5, 1, 4)
ax5 = fig.add_subplot(5, 1, 5)

# Graph for x1 (arm angle)
ax1.plot(time, x1, color="blue", lw=2)
ax1.set_xlim([0, T_end])
ax1.set_ylabel("θ_a [deg]")
ax1.set_xlabel("time[sec]")

# Graph for x2 (pendulum angle)
ax2.plot(time, x2, color="blue", lw=2)
ax2.set_xlim([0, T_end])
ax2.set_ylabel("Φ_p [deg]")
ax2.set_xlabel("time[sec]")

# Graph for x3 (arm angle velocity)
ax3.plot(time, x3, color="blue", lw=2)
ax3.set_xlim([0, T_end])
ax3.set_ylabel("dot_θ_a [deg/sec]")
ax3.set_xlabel("time[sec]")

# Graph for x4 (pendulum angle velocity)
ax4.plot(time, x4, color="blue", lw=2)
ax4.set_xlim([0, T_end])
ax4.set_ylabel("dot_Φ_p [deg/sec]")
ax4.set_xlabel("time[sec]")

# Graph for V (input voltage)
ax5.plot(time, V, color="blue", lw=2)
ax5.set_xlim([0, T_end])
ax5.set_ylabel("Voltage[V]")
ax5.set_xlabel("time[sec]")

# plt.tight_layout()
plt.show()
