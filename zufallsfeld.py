import numpy as np
import matplotlib.pyplot as plt

# Parameter
n = 200
x = np.linspace(0, 5, n)

def C1(x, y, v, s, p):
    return v * np.exp(-s * np.abs(x - y)**p)

def C2(x, y, v, s, p, sigma):
    return C1(x, y, v, s, p) + sigma * (x == y)

def simulate_field(v=1.0, s=1.0, p=1.0, sigma=0.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X, Y = np.meshgrid(x, x)
    K = C1(X, Y, v, s, p)
    if sigma > 0:
        K += sigma * np.eye(n)
    L = np.linalg.cholesky(K + 1e-10*np.eye(n))
    g = np.random.randn(n)
    Z = L @ g
    return Z

ps = [0.5, 1.0, 1.5, 1.95]
v_list = [0.5, 1.0]
s_list = [1.0, 2.0]
sigma_list = [0.05, 0.2]

# C1 figure
fig1, axs1 = plt.subplots(len(ps), len(v_list), figsize=(10, 2.5*len(ps)), sharex=True, sharey=True)
for i, p in enumerate(ps):
    for j, (v, s) in enumerate(zip(v_list, s_list)):
        Z1 = simulate_field(v=v, s=s, p=p, sigma=0.0, seed=42)
        axs1[i, j].plot(x, Z1, label=f"C1 v={v}, s={s}")
        axs1[i, j].set_title(f"C1: p={p}, v={v}, s={s}")
        axs1[i, j].legend()
        axs1[i, j].grid(True, alpha=0.3)
plt.tight_layout()
fig1.savefig("C1_fields.png")

# C2 figure
fig2, axs2 = plt.subplots(len(ps), len(sigma_list), figsize=(10, 2.5*len(ps)), sharex=True, sharey=True)
for i, p in enumerate(ps):
    for j, (v, s, sigma) in enumerate(zip(v_list, s_list, sigma_list)):
        Z2 = simulate_field(v=v, s=s, p=p, sigma=sigma, seed=42)
        axs2[i, j].plot(x, Z2, label=f"C2 σ={sigma}")
        axs2[i, j].set_title(f"C2: p={p}, v={v}, s={s}, σ={sigma}")
        axs2[i, j].legend()
        axs2[i, j].grid(True, alpha=0.3)
plt.tight_layout()
fig2.savefig("C2_fields.png")

# Ergebnisse der Simulationen:
#
# C1-Kovarianzfunktion: C1(x,y) = v * exp(-s * |x-y|^p)
# - p = 1.95: sehr glatte Felder mit starker räumlicher Korrelation
# - p = 1.5: mittlere Glättheit
# - p = 1.0: weniger glatte Felder
# - p = 0.5: sehr viele lokale Schwankungen, fast unkorreliert
#
# Parametereffekte:
# - Größeres v: Höhere Varianz -> größere Amplituden der Zufallsfelder
# - Größeres s: Schnellerer Korrelationsabfall -> weniger glatte Bewegungen
#
# C2-Kovarianzfunktion: C2(x,y) = C1(x,y) + σ * δ_{x,y} (Nugget-Effekt)
# - σ > 0 fügt unkorreliertes Rauschen hinzu -> "rauere" Felder mit Diskontinuitäten
# - Je größer σ, desto stärker der Nugget-Effekt und weniger glatt die Realisierungen
#
# Die Figuren zeigen:
# - C1_fields.png: Glatte Zufallsfelder, Glättheit nimmt mit steigendem p zu
# - C2_fields.png: Zusätzliches Rauschen durch Nugget-Effekt