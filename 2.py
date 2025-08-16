#!/usr/bin/env python3
import numpy as np
import first as f
import matplotlib.pyplot as plt


def compute_error(exper, theory, f):
    val_exp = f(exper)
    val_theory = f(theory)

    return np.abs(val_exp - val_theory) / val_theory


# f (x, y) = x⁴ + y⁴ − 4xy + (1 / 2) y + 1.
def a_func(vars):
    x = vars[0]
    y = vars[1]
    return x**4 + y**4 - 4 * x * y + 0.5 * y + 1


def a_grad(vars):
    x = vars[0]
    y = vars[1]
    return np.array([4 * (x**3) - 4 * y, 4 * (y**3) - 4 * x + 0.5])


def a_hess(vars):
    x = vars[0]
    y = vars[1]
    return np.array([[12 * (x**2), 4], [-4, 12 * (y**2)]])


cases = {
    "a": {
        "func": a_func,
        "grad": a_grad,
        "hess": a_hess,
        "x0": np.array([-3, 1]),
        "alpha": 5e-2,
        "max_iter": 5000,
        "epsilon": 1e-6,
        "optimum": np.array([-1.01463, -1.04453]),
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
    for name, method in methods.items():
        try:
            result = method(
                case["func"],
                case["grad"],
                case["hess"],
                case["x0"],
                case["alpha"],
                case["max_iter"],
                case["epsilon"],
            )
            resultTable.append(
                {
                    "name": name,
                    "converged": result["converged"],
                    "iterations": result["iterations"],
                    "solution": result["best"],
                    "error": f"{compute_error(result["best"], case["optimum"], case["func"]):.6e}",
                }
            )

            # print("Result x_sequence", result["x_sequence"])

            if name == "Polak-Ribière" or case == "c":
                continue
            if key == "a" or key == "b":
                x = [x[0] for x in result["x_sequence"]]
                y = [x[1] for x in result["x_sequence"]]

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
            print(f"\n{name}: Error - {e}")
    # Add labels and title (optional but recommended)

    # ax.title(f"Plot for {key}")

    titles = {
        "Algoritmo": 20,
        "Convergió": 20,
        "Iteraciones": 20,
        "Solución": 40,
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
                print(f"{fieldValue}".center(width, " "), end="|")
        print()

    if case != "c":
        x_best = [case["optimum"][0]]
        y_best = [case["optimum"][1]]
        # z_best = [case["func"](case["optimum"])]

        plt.plot(x_best, y_best, label="Solución")
        # ax.scatter(x_best, y_best, z_best, label="Solución")

        # Define the range for x and y
        x = np.linspace(min_x, max_x, int(max_x - min_x))
        y = np.linspace(min_y, max_y, int(max_y - min_y))

        # Create a meshgrid
        X, Y = np.meshgrid(x, y)

        # Evaluate the function on the grid
        Z = case["func"]([X, Y])

        plt.legend(title="Algoritmo")
        plt.xlabel("X")
        plt.ylabel("Y")
        # ax.set_zlabel("f(X,Y)")
        plt.contour(X, Y, Z, levels=int(max_x - min_x), cmap="viridis")

        # Display the plot
        plt.show()
