import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Parametre
# -----------------------------
g = 9.81
ell = 0.10
m = 0.10

omega0 = np.sqrt(g / ell)
t_max = 10.0


# -----------------------------
# Modell
# -----------------------------
def f(theta, omega):
    dtheta = omega
    domega = -(g / ell) * np.sin(theta)
    return dtheta, domega


# -----------------------------
# Eksplisitt Euler
# -----------------------------
def euler(theta0, omega0_init, dt):
    t = np.arange(0, t_max + dt, dt)
    theta = np.zeros_like(t)
    omega = np.zeros_like(t)

    theta[0] = theta0
    omega[0] = omega0_init

    for n in range(len(t) - 1):
        dtheta, domega = f(theta[n], omega[n])
        theta[n + 1] = theta[n] + dt * dtheta
        omega[n + 1] = omega[n] + dt * domega

    return t, theta, omega


# -----------------------------
# Energier
# -----------------------------
def energies(theta, omega):
    T = 0.5 * m * (ell**2) * omega**2
    V = m * g * ell * (1 - np.cos(theta))
    E = T + V
    return T, V, E


# ============================================================
# (a) θ(t)/θ0 for ulike initialutslag
# ============================================================
theta0_list = [0.1, 0.5, 1.0]
dt = 1e-3

plt.figure()
for theta0 in theta0_list:
    t, theta, omega = euler(theta0, 0.0, dt)
    plt.plot(t, theta / theta0, label=f"θ0 = {theta0}")

# Analytisk småvinkel
t_ana = np.linspace(0, t_max, 2000)
plt.plot(t_ana, np.cos(omega0 * t_ana), "k--", label="cos(ω0 t)")

plt.xlabel("t [s]")
plt.ylabel("θ(t)/θ0")
plt.title("(a) Normalisert vinkelutslag")
plt.legend()
plt.grid()
plt.show()      # <-- VIKTIG: lukk dette vinduet for å få (b)


# ============================================================
# (b) Energier for θ0 = 0.5 rad og ulike Δt
# ============================================================
theta0 = 0.5
dt_list = [1e-3, 1e-2, 1e-1]

for dt in dt_list:
    t, theta, omega = euler(theta0, 0.0, dt)
    T, V, E = energies(theta, omega)

    plt.figure()
    plt.plot(t, T, label="T(t) kinetisk")
    plt.plot(t, V, label="V(t) potensiell")
    plt.plot(t, E, label="E(t) total")

    plt.xlabel("t [s]")
    plt.ylabel("Energi [J]")
    plt.title(f"(b) Energier, θ0=0.5 rad, Δt={dt}")
    plt.legend()
    plt.grid()
    plt.show()

    # Energidrift
    rel_drift = (E - E[0]) / abs(E[0])
    print(f"Δt = {dt:>6}: maks relativ energidrift = {np.max(np.abs(rel_drift)):.2e}")
