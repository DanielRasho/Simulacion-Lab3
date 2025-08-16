#!/usr/bin/env python3

import numpy as np
from typing import Callable, Tuple, List, Dict, Any


def gradient_descent_random_direction(
    f: Callable,
    df: Callable,
    ddf: Callable,
    x0: np.ndarray,
    alpha: float,
    max_iter: int,
    epsilon: float,
) -> Dict[str, Any]:
    """
    Descenso gradiente naïve con dirección de descenso aleatoria
    """
    x = x0.copy()
    n = len(x0)

    x_sequence = [x.copy()]
    f_sequence = [f(x)]
    errors = []

    for k in range(max_iter):
        grad = df(x)

        # Criterio de paro: norma del gradiente
        error = np.linalg.norm(grad)
        errors.append(error)

        if error < epsilon:
            converged = True
            break

        # Dirección aleatoria (vector unitario aleatorio)
        d = np.random.randn(n)
        d = d / np.linalg.norm(d)

        # Asegurar que sea dirección de descenso
        if np.dot(grad, d) > 0:
            d = -d

        x_new = x - alpha * d

        x_sequence.append(x_new.copy())
        f_sequence.append(f(x_new))

        x = x_new
    else:
        converged = False

    return {
        "best": x,
        "x_sequence": x_sequence,
        "f_sequence": f_sequence,
        "errors": errors,
        "iterations": len(errors),
        "converged": converged,
    }


def steepest_descent_naive(
    f: Callable,
    df: Callable,
    ddf: Callable,
    x0: np.ndarray,
    alpha: float,
    max_iter: int,
    epsilon: float,
) -> Dict[str, Any]:
    """
    Descenso máximo naïve (steepest descent)
    """
    x = x0.copy()

    x_sequence = [x.copy()]
    f_sequence = [f(x)]
    errors = []

    for k in range(max_iter):
        grad = df(x)

        # Criterio de paro: norma del gradiente
        error = np.linalg.norm(grad)
        errors.append(error)

        if error < epsilon:
            converged = True
            break

        # Dirección de descenso más pronunciada (negativo del gradiente)
        d = -grad

        x_new = x + alpha * d

        x_sequence.append(x_new.copy())
        f_sequence.append(f(x_new))

        x = x_new
    else:
        converged = False

    return {
        "best": x,
        "x_sequence": x_sequence,
        "f_sequence": f_sequence,
        "errors": errors,
        "iterations": len(errors),
        "converged": converged,
    }


def newton_method(
    f: Callable,
    df: Callable,
    ddf: Callable,
    x0: np.ndarray,
    alpha: float,
    max_iter: int,
    epsilon: float,
) -> Dict[str, Any]:
    """
    Método de Newton con Hessiano exacto
    """
    x = x0.copy()

    x_sequence = [x.copy()]
    f_sequence = [f(x)]
    errors = []

    for k in range(max_iter):
        grad = df(x)
        hess = ddf(x)

        # Criterio de paro: norma del gradiente
        error = np.linalg.norm(grad)
        errors.append(error)

        if error < epsilon:
            converged = True
            break

        try:
            # Dirección de Newton
            d = -np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            # Si el Hessiano es singular, usar gradiente
            d = -grad

        x_new = x + alpha * d

        x_sequence.append(x_new.copy())
        f_sequence.append(f(x_new))

        x = x_new
    else:
        converged = False

    return {
        "best": x,
        "x_sequence": x_sequence,
        "f_sequence": f_sequence,
        "errors": errors,
        "iterations": len(errors),
        "converged": converged,
    }


def conjugate_gradient_fletcher_reeves(
    f: Callable,
    df: Callable,
    ddf: Callable,
    x0: np.ndarray,
    alpha: float,
    max_iter: int,
    epsilon: float,
) -> Dict[str, Any]:
    """
    Método de gradiente conjugado Fletcher-Reeves
    """
    x = x0.copy()
    grad = df(x)
    d = -grad.copy()  # Dirección inicial

    x_sequence = [x.copy()]
    f_sequence = [f(x)]
    errors = []

    for k in range(max_iter):
        grad = df(x)

        # Criterio de paro: norma del gradiente
        error = np.linalg.norm(grad)
        errors.append(error)

        if error < epsilon:
            converged = True
            break

        # Actualizar posición
        x_new = x + alpha * d
        x_sequence.append(x_new.copy())
        f_sequence.append(f(x_new))

        # Calcular nuevo gradiente
        grad_new = df(x_new)

        # Parámetro beta de Fletcher-Reeves
        beta = np.dot(grad_new, grad_new) / np.dot(grad, grad)

        # Nueva dirección conjugada
        # print(f"d = -{grad_new} + {beta} * {d}")
        d = -grad_new + beta * d

        x = x_new
        grad = grad_new
    else:
        converged = False

    return {
        "best": x,
        "x_sequence": x_sequence,
        "f_sequence": f_sequence,
        "errors": errors,
        "iterations": len(errors),
        "converged": converged,
    }


def conjugate_gradient_polak_ribiere(
    f: Callable,
    df: Callable,
    ddf: Callable,
    x0: np.ndarray,
    alpha: float,
    max_iter: int,
    epsilon: float,
) -> Dict[str, Any]:
    """
    Método de gradiente conjugado Polak-Ribière
    """
    x = x0.copy()
    grad = df(x)
    d = -grad.copy()  # Dirección inicial

    x_sequence = [x.copy()]
    f_sequence = [f(x)]
    errors = []

    for k in range(max_iter):
        grad = df(x)

        # Criterio de paro: norma del gradiente
        error = np.linalg.norm(grad)
        errors.append(error)

        if error < epsilon:
            converged = True
            break

        # Actualizar posición
        x_new = x + alpha * d
        x_sequence.append(x_new.copy())
        f_sequence.append(f(x_new))

        # Calcular nuevo gradiente
        grad_new = df(x_new)

        # Parámetro beta de Polak-Ribière
        y = grad_new - grad
        beta = np.dot(grad_new, y) / np.dot(grad, grad)
        beta = max(0, beta)  # Reinicio automático

        # Nueva dirección conjugada
        d = -grad_new + beta * d

        x = x_new
        grad = grad_new
    else:
        converged = False

    return {
        "best": x,
        "x_sequence": x_sequence,
        "f_sequence": f_sequence,
        "errors": errors,
        "iterations": len(errors),
        "converged": converged,
    }


def bfgs_method(
    f: Callable,
    df: Callable,
    ddf: Callable,
    x0: np.ndarray,
    alpha: float,
    max_iter: int,
    epsilon: float,
) -> Dict[str, Any]:
    """
    Método BFGS (Broyden-Fletcher-Goldfarb-Shanno)
    """
    x = x0.copy()
    n = len(x0)

    # Inicializar aproximación del Hessiano inverso
    H = np.eye(n)

    x_sequence = [x.copy()]
    f_sequence = [f(x)]
    errors = []

    grad = df(x)

    for k in range(max_iter):
        grad = df(x)

        # Criterio de paro: norma del gradiente
        error = np.linalg.norm(grad)
        errors.append(error)

        if error < epsilon:
            converged = True
            break

        # Dirección de búsqueda
        d = -H @ grad

        # Actualizar posición
        x_new = x + alpha * d
        x_sequence.append(x_new.copy())
        f_sequence.append(f(x_new))

        # Calcular nuevo gradiente
        grad_new = df(x_new)

        # Vectores para actualización BFGS
        s = x_new - x
        y = grad_new - grad

        # Evitar división por cero
        sy = np.dot(s, y)
        if abs(sy) > 1e-10:
            # Actualización BFGS del Hessiano inverso
            rho = 1.0 / sy

            # Fórmula de Sherman-Morrison
            I = np.eye(n)
            V = I - rho * np.outer(s, y)
            H = V @ H @ V.T + rho * np.outer(s, s)

        x = x_new
        grad = grad_new
    else:
        converged = False

    return {
        "best": x,
        "x_sequence": x_sequence,
        "f_sequence": f_sequence,
        "errors": errors,
        "iterations": len(errors),
        "converged": converged,
    }


# Ejemplo de uso con la función de Rosenbrock
def rosenbrock(x):
    """Función de Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²"""
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2


def rosenbrock_grad(x):
    """Gradiente de la función de Rosenbrock"""
    return np.array(
        [-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2), 200 * (x[1] - x[0] ** 2)]
    )


def rosenbrock_hess(x):
    """Hessiano de la función de Rosenbrock"""
    return np.array(
        [[-2 + 1200 * x[0] ** 2 - 400 * x[1], -400 * x[0]], [-400 * x[0], 200]]
    )


# Ejemplo de prueba
if __name__ == "__main__":
    # Parámetros
    x0 = np.array([-1.2, 1.0])
    alpha = 0.001
    max_iter = 1000
    epsilon = 1e-6

    print("Comparación de métodos de descenso gradiente")
    print("=" * 50)

    methods = {
        "Dirección aleatoria": gradient_descent_random_direction,
        "Steepest Descent": steepest_descent_naive,
        "Newton": newton_method,
        "Fletcher-Reeves": conjugate_gradient_fletcher_reeves,
        "Polak-Ribière": conjugate_gradient_polak_ribiere,
        "BFGS": bfgs_method,
    }

    for name, method in methods.items():
        try:
            result = method(
                rosenbrock,
                rosenbrock_grad,
                rosenbrock_hess,
                x0,
                alpha,
                max_iter,
                epsilon,
            )
            print(f"\n{name}:")
            print(f"  Mejor solución: {result['best']}")
            print(f"  Valor función: {result['f_sequence'][-1]:.6f}")
            print(f"  Iteraciones: {result['iterations']}")
            print(f"  Convergencia: {result['converged']}")
            print(f"  Error final: {result['errors'][-1]:.6e}")
        except Exception as e:
            print(f"\n{name}: Error - {e}")
