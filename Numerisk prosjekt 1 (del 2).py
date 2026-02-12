import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parametre (endre om oppgaven gir andre tall)
# -----------------------------
g = 9.81
ell = 0.10     # pendellengde
m = 0.10       # masse per pendel
M = 1.0        # bjelkemasse
Delta = 0.20   # avstand mellom hengslene (påvirker x_cm offset, men ikke dynamikken)

t_max = 30.0
dt = 1e-3      # prøv 1e-3, evt 2e-3 eller 5e-4 for bedre energi

alpha = m / (M + 2*m)

# -----------------------------
# Hjelpefunksjoner
# -----------------------------
def xb_dot(theta1, theta2, w1, w2):
    """Horisontalhastighet til bjelken fra horisontal impulsbevaring (start i ro)."""
    return -(m * ell / (M + 2*m)) * (np.cos(theta1)*w1 + np.cos(theta2)*w2)

def wdot_from_thetas(theta1, theta2, w1, w2):
    """
    Løser 2x2-systemet for [theta1_ddot, theta2_ddot] (dvs. [w1_dot, w2_dot]).
    Basert på:
      (1 - alpha cos^2 t1) t1dd - alpha cos t1 cos t2 t2dd = -(g/ell) sin t1 + alpha cos t1 S
      -alpha cos t1 cos t2 t1dd + (1 - alpha cos^2 t2) t2dd = -(g/ell) sin t2 + alpha cos t2 S
    der S = sin t1 w1^2 + sin t2 w2^2
    """
    c1, c2 = np.cos(theta1), np.cos(theta2)
    s1, s2 = np.sin(theta1), np.sin(theta2)

    S = s1*(w1**2) + s2*(w2**2)

    a11 = 1.0 - alpha*(c1**2)
    a22 = 1.0 - alpha*(c2**2)
    a12 = -alpha*c1*c2
    A = np.array([[a11, a12],
                  [a12, a22]], dtype=float)

    rhs1 = -(g/ell)*s1 + alpha*c1*S
    rhs2 = -(g/ell)*s2 + alpha*c2*S
    rhs = np.array([rhs1, rhs2], dtype=float)

    tdd = np.linalg.solve(A, rhs)
    return tdd[0], tdd[1]

def deriv(t, y):
    """
    y = [theta1, theta2, w1, w2, xb]
    Returnerer y' = [w1, w2, w1dot, w2dot, xbdot]
    """
    theta1, theta2, w1, w2, xb = y
    w1dot, w2dot = wdot_from_thetas(theta1, theta2, w1, w2)
    xbd = xb_dot(theta1, theta2, w1, w2)
    return np.array([w1, w2, w1dot, w2dot, xbd], dtype=float)

def rk4_step(t, y, h):
    k1 = deriv(t, y)
    k2 = deriv(t + 0.5*h, y + 0.5*h*k1)
    k3 = deriv(t + 0.5*h, y + 0.5*h*k2)
    k4 = deriv(t + h, y + h*k3)
    return y + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)

def energies(theta1, theta2, w1, w2):
    """
    Total energi:
      KE = 1/2 M xbdot^2 + sum_i 1/2 m vi^2
      vi^2 = (xbdot + ell cos(theta_i) wi)^2 + (ell sin(theta_i) wi)^2
          = xbdot^2 + 2 xbdot ell cos(theta_i) wi + ell^2 wi^2
      PE = sum_i m g ell (1 - cos(theta_i))
    """
    xbd = xb_dot(theta1, theta2, w1, w2)

    c1, c2 = np.cos(theta1), np.cos(theta2)
    # speed^2 for each bob
    v1_sq = xbd**2 + 2*xbd*ell*c1*w1 + (ell**2)*(w1**2)
    v2_sq = xbd**2 + 2*xbd*ell*c2*w2 + (ell**2)*(w2**2)

    T = 0.5*M*(xbd**2) + 0.5*m*v1_sq + 0.5*m*v2_sq
    V = m*g*ell*(1 - np.cos(theta1)) + m*g*ell*(1 - np.cos(theta2))
    return T, V, T+V

def x_center_of_mass(xb, theta1, theta2):
    """
    x_cm = (M xb + m x1 + m x2)/(M+2m)
    x1 = xb - Delta/2 + ell sin(theta1)
    x2 = xb + Delta/2 + ell sin(theta2)
    Delta-korrigeringen kansellerer faktisk i summen, men vi lar den stå eksplisitt.
    """
    x1 = xb - Delta/2 + ell*np.sin(theta1)
    x2 = xb + Delta/2 + ell*np.sin(theta2)
    return (M*xb + m*x1 + m*x2) / (M + 2*m)

# -----------------------------
# Initialtilstand
# -----------------------------
theta1_0 = 0.5
theta2_0 = 0.0
w1_0 = 0.0
w2_0 = 0.0
xb_0 = 0.0

# -----------------------------
# Simulering
# -----------------------------
t = np.arange(0.0, t_max + dt, dt)
y = np.zeros((t.size, 5), dtype=float)
y[0] = np.array([theta1_0, theta2_0, w1_0, w2_0, xb_0], dtype=float)

for n in range(t.size - 1):
    y[n+1] = rk4_step(t[n], y[n], dt)

theta1 = y[:, 0]
theta2 = y[:, 1]
w1 = y[:, 2]
w2 = y[:, 3]
xb = y[:, 4]

# -----------------------------
# Energi og massensenter
# -----------------------------
T = np.zeros_like(t)
V = np.zeros_like(t)
E = np.zeros_like(t)
xcm = np.zeros_like(t)

for i in range(t.size):
    Ti, Vi, Ei = energies(theta1[i], theta2[i], w1[i], w2[i])
    T[i], V[i], E[i] = Ti, Vi, Ei
    xcm[i] = x_center_of_mass(xb[i], theta1[i], theta2[i])

# Relativ energidrift (nyttig “nøyaktighetssjekk”)
rel_drift = (E - E[0]) / (abs(E[0]) if abs(E[0]) > 0 else 1.0)
print(f"Maks relativ energidrift over {t_max}s (dt={dt}): {np.max(np.abs(rel_drift)):.3e}")
print(f"x_cm-endring (maks |x_cm - x_cm(0)|): {np.max(np.abs(xcm - xcm[0])):.3e} m")

# -----------------------------
# Plott
# -----------------------------
plt.figure()
plt.plot(t, E, label="E(t) total")
plt.plot(t, T, label="T(t) kinetisk")
plt.plot(t, V, label="V(t) potensiell")
plt.xlabel("t [s]")
plt.ylabel("Energi [J]")
plt.title("Energi vs tid (to pendler på fri bjelke)")
plt.grid(True)
plt.legend()
plt.show()

plt.figure()
plt.plot(t, xcm, label="x_cm(t)")
plt.xlabel("t [s]")
plt.ylabel("x_cm [m]")
plt.title("Massensenterets x-posisjon vs tid")
plt.grid(True)
plt.legend()
plt.show()


