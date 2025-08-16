#!/usr/bin/env python3
import numpy as np
import first as f
import math as m
import traceback
import matplotlib.pyplot as plt


def has_nan(arr):
    for a in arr:
        if m.isnan(a):
            return True
    return False


def compute_error(exper, theory, f):
    val_exp = f(exper)
    val_theory = f(theory)

    return np.abs(val_exp - val_theory) / (1 if val_theory == 0 else val_theory)


# f (x, y) = x⁴ + y⁴ − 4xy + (1 / 2) y + 1.
def a_func(vars):
    x = vars[0]
    y = vars[1]
    return x**4 + y**4 - 4 * x * y + 0.5 * y + 1


def a_grad(vars):
    x = vars[0]
    y = vars[1]
    return np.array(
        [
            4 * (x**3) - 4 * y,
            4 * (y**3) - 4 * x + 0.5,
        ]
    )


def a_hess(vars):
    x = vars[0]
    y = vars[1]
    return np.array(
        [
            [12 * (x**2), 4],
            [-4, 12 * (y**2)],
        ]
    )


# f(x1, x2) = 100*(x2 − x1²)² + (1 − x1)²
def b_func(vars):
    x1 = vars[0]
    x2 = vars[1]
    return 100 * ((x2 - x1**2) ** 2) + (1 - x1) ** 2


def b_grad(vars):
    x = vars[0]
    y = vars[1]
    # Extended form: 100y^2-200yx^2+100x^4+1-2x+x^2
    return np.array(
        [
            -400 * y * x + 400 * (x**3) - 2 + 2 * x,
            200 * y - 200 * (x**2),
        ]
    )


def b_hess(vars):
    x = vars[0]
    y = vars[1]
    return np.array(
        [
            [-400 * y + 1200 * (x**2) + 2, -400 * x],
            [-400 * x, 200],
        ]
    )


def c_func(x):
    return sum(
        [100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1)]
    )


def c_grad(x):
    # 100 * (y - x²)² + (1-x)²
    # 100 * (y - x²)² + 1-2x+x²
    # 100 * (y²-2yx²-x⁴) + 1-2x+x²
    # 100y²-200yx²-100x⁴ + 1-2x+x²

    # If we roll-out each iteration:
    #   100x2²-200x2x1²-100x1⁴ + 1-2x1+x1²
    # + 100x3²-200x3x2²-100x2⁴ + 1-2x2+x2²
    # + 100x4²-200x4x3²-100x3⁴ + 1-2x3+x3²
    # + 100x5²-200x5x4²-100x4⁴ + 1-2x4+x4²
    # + 100x6²-200x6x5²-100x5⁴ + 1-2x5+x5²
    # + 100x7²-200x7x6²-100x6⁴ + 1-2x6+x6²

    # Now we derive:
    # x1: -400x2x1-400x1³-2+2x1
    # x2: 200x2-200x1²-400x3x2-400x2³-2+2x2
    # x3: 200x3-200x2²-400x4x3-400x3³-2+2x3
    # ...
    # x7: 200x7
    grad = []
    grad.append(-400 * x[1] * x[0] - 400 * x[0] ** 3 - 2 + 2 * x[0])
    for i in range(1, 6):
        grad.append(
            200 * x[i]
            - 200 * x[i - 1]
            - 400 * x[i + 1] * x[i]
            - 400 * x[i] ** 3
            - 2
            + 2 * x[i]
        )
    grad.append(200 * x[6])
    return np.array(grad)


def c_hess(x):
    # Define all derivations:
    # x1: -400x2x1-400x1³-2+2x1
    # x2: 200x2-200x1²-400x3x2-400x2³-2+2x2
    # x3: 200x3-200x2²-400x4x3-400x3³-2+2x3
    # x4: 200x4-200x3²-400x5x4-400x4³-2+2x4
    # x5: 200x5-200x4²-400x6x5-400x5³-2+2x5
    # x6: 200x6-200x5²-400x7x6-400x6³-2+2x6
    # x7: 200x7

    hess = []
    hess.append([-400 * x[1] - 1200 * x[0] ** 2 + 1, -400 * x[0], 0, 0, 0, 0, 0])
    for i in range(1, 6):
        row = []
        for j in range(7):
            if j == i - 1:
                row.append(-400 * x[j])
            elif j == i:
                row.append(200 - 400 * x[i + 1] - 1200 * x[i] ** 2 + 2)
            elif j == i + 1:
                row.append(-400 * x[i])
            else:
                row.append(0)
        hess.append(row)

    hess.append([0, 0, 0, 0, 0, 0, 200])

    return np.array(hess)


cases = {
    "a": {
        "func": a_func,
        "grad": a_grad,
        "hess": a_hess,
        "x0": np.array([-3, 1]),
        "alpha": [9e-6, 5e-2, 5e-2, 5e-2, 85e-5, 5e-2],
        "max_iter": 5000,
        "epsilon": 1e-6,
        "optimum": np.array([-1.01463, -1.04453]),
    },
    "b": {
        "func": b_func,
        "grad": b_grad,
        "hess": b_hess,
        "x0": np.array([-1.2, 1.0]),
        "alpha": [1e-3, 1e-3, 1e-3, 1e-9, 1e-3, 5e-2],
        "max_iter": 10000,
        "epsilon": 1e-2,
        "optimum": np.array([1.0, 1.0]),
    },
    "c": {
        "func": c_func,
        "grad": c_grad,
        "hess": c_hess,
        "x0": np.array([-1.2, 1, 1, 1, 1, -1.2, 1]),
        "alpha": [5e-3, 1e-10, 2.5e-2, 1e-12, 1e-8, 1e-6],
        "max_iter": 10000,
        "epsilon": 1e-3,
        "optimum": np.array([1, 1, 1, 1, 1, 1, 1]),
    },
}

for key, case in cases.items():
    print(f" Evaluando caso {key} ".center(150, "="))
    methods = {
        "Dirección aleatoria": f.gradient_descent_random_direction,
        "Steepest Descent": f.steepest_descent_naive,
        "Newton": f.newton_method,
        "Fletcher-Reeves": f.conjugate_gradient_fletcher_reeves,
        "Polak-Ribière": f.conjugate_gradient_polak_ribiere,
        "BFGS": f.bfgs_method,
    }
    min_x = float("+inf")
    max_x = float("-inf")

    min_y = float("+inf")
    max_y = float("-inf")

    # fig = plt.figure()
    resultTable = []
    for idx, (name, method) in enumerate(methods.items()):
        print("Usando método:", name)
        try:
            result = method(
                case["func"],
                case["grad"],
                case["hess"],
                case["x0"],
                case["alpha"][idx],
                case["max_iter"],
                case["epsilon"],
            )
            resultTable.append(
                {
                    "name": name,
                    "converged": result["converged"],
                    "iterations": result["iterations"],
                    "solution": result["best"],
                    "error": f"{compute_error(result["best"], case["optimum"], case["func"]):.2%}",
                }
            )

            # print("Result x_sequence", result["x_sequence"])

            if case == "c":
                continue
            if key == "a" or key == "b":
                x = [x[0] for x in result["x_sequence"]]
                if has_nan(x):
                    print(f"Ignoring {name} because it has NaN!")
                    continue

                y = [x[1] for x in result["x_sequence"]]
                if has_nan(y):
                    print(f"Ignoring {name} because it has NaN!")
                    continue

                local_min_x = min(x)
                local_max_x = max(x)
                if local_min_x < min_x:
                    min_x = local_min_x
                if local_max_x > max_x:
                    max_x = local_max_x

                local_min_y = min(y)
                local_max_y = max(y)
                if local_min_y < min_y:
                    min_y = local_min_y
                if local_max_y > max_y:
                    max_y = local_max_y

                # print(x)

                # Create the plot
                # ax.stem(x, y, z, label=name)
                # ax.scatter(x, y, z, label=name)
                plt.plot(x, y, label=name)

            # print(f"\n{name}:")
            # print(f"  Mejor solución: {result['best']}")
            # print(f"  Valor función: {result['f_sequence'][-1]:.6f}")
            # print(f"  Iteraciones: {result['iterations']}")
            # print(f"  Convergencia: {result['converged']}")
            # print(f"  Error final: {result['errors'][-1]:.6e}")
        except Exception as e:
            print(f"{name}: Error - {e}")
            traceback.print_exc()
    # Add labels and title (optional but recommended)

    # ax.title(f"Plot for {key}")

    titles = {
        "Algoritmo": 20,
        "Convergió": 20,
        "Iteraciones": 20,
        "Solución": 40 if key != "c" else 120,
        "Error": 15,
    }
    for title, width in titles.items():
        print(title.center(width, " "), end="|")
    print()
    for title, width in titles.items():
        print("-" * width, end="|")
    print()

    for row in resultTable:
        for (fieldName, fieldValue), width in zip(row.items(), titles.values()):
            if fieldName == "name":
                print(fieldValue.ljust(width, " "), end="|")
            else:
                print(f"{fieldValue}".replace("\n", "").center(width, " "), end="|")
        print()

    if key != "c":
        x_best = [case["optimum"][0]]
        y_best = [case["optimum"][1]]
        # z_best = [case["func"](case["optimum"])]

        plt.plot(x_best, y_best, label="Solución")
        # ax.scatter(x_best, y_best, z_best, label="Solución")

        # Define the range for x and y
        x = np.linspace(min_x, max_x, int(min(max_x - min_x, 100)))
        y = np.linspace(min_y, max_y, int(min(max_y - min_y, 100)))

        # Create a meshgrid
        X, Y = np.meshgrid(x, y)

        # Evaluate the function on the grid
        Z = case["func"]([X, Y])

        plt.legend(title="Algoritmo")
        plt.xlabel("X")
        plt.ylabel("Y")
        # ax.set_zlabel("f(X,Y)")
        plt.contour(X, Y, Z, levels=int(min(max_x - min_x, 50)), cmap="viridis")

        # Display the plot
        plt.show()

    print("Solución óptima:", case["optimum"])
