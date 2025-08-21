import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


data = pd.read_csv("datos_lab3.csv")
x = data["x"].values
y = data["y"].values
n = len(x)


def f(x, beta):
    return beta[0] + beta[1]*x + beta[2]*x**2 + beta[3]*np.sin(7*x) + beta[4]*np.sin(13*x)

def error(beta, lam):
    # error cuadrático
    mse = np.sum((f(x, beta) - y)**2)
    # término de suavizado
    smooth = np.sum((f(x[1:], beta) - f(x[:-1], beta))**2)
    return mse + lam * smooth


lambdas = [0, 100, 500]
betas = {}

for lam in lambdas:
    beta0 = np.zeros(5)  # inicialización
    res = minimize(error, beta0, args=(lam,), method="BFGS")
    betas[lam] = res.x
    print(f"λ={lam} → β={res.x}")


x_grid = np.linspace(min(x), max(x), 500)

plt.scatter(x, y, color="black", label="Datos")
for lam in lambdas:
    plt.plot(x_grid, f(x_grid, betas[lam]), label=f"λ={lam}")
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.title("Regresión con y sin regularización")
plt.show()
