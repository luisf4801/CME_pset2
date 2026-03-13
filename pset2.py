# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:05:06 2026

@author: Luis
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 17:36:03 2026

@author: Luis
"""

#### Pacotes
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, Akima1DInterpolator
from numpy.polynomial import polynomial as P
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
os.makedirs("output", exist_ok=True)

np.random.seed(42)

###### Q1

df = pd.DataFrame({
    "ano": [1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010],
    "expectativa_de_vida": [45.5, 48.0, 52.5, 57.6, 62.5, 66.9, 71.1, 74.4]
})

x = df["ano"].values
y = df["expectativa_de_vida"].values
x_dense = np.linspace(x.min(), x.max(), 300)

# 1. linear entre pontos (piecewise)
y_piecewise = np.interp(x_dense, x, y)

# 2. regressão linear única
coef = np.polyfit(x, y, 1)
y_linear = np.polyval(coef, x_dense)

# 3. cubic spline
cs = CubicSpline(x, y)
y_spline = cs(x_dense)

# plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_dense, y_piecewise, color="steelblue", lw=2,  label="Linear entre pontos")
ax.plot(x_dense, y_linear,   color="tomato",     lw=2,  label="Regressão linear", linestyle="--")
ax.plot(x_dense, y_spline,   color="seagreen",   lw=2,  label="Cubic spline")
ax.scatter(x, y,             color="steelblue",  s=70,  zorder=5, label="Dados originais")
ax.set_xlabel("Ano")
ax.set_ylabel("Expectativa de vida (anos)")
ax.set_title("Interpolação da expectativa de vida global")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("output/plot_function_expectativa_de_vida.png", dpi=150, bbox_inches="tight")
plt.show()

ano_query = 1996

estimativas = [
    ["Linear entre pontos", f"{np.interp(ano_query, x, y):.2f}"],
    ["Regressão linear",    f"{np.polyval(coef, ano_query):.2f}"],
    ["Cubic spline",        f"{float(cs(ano_query)):.2f}"],
]

fig, ax = plt.subplots(figsize=(6, 2.5))
ax.axis("off")

columns     = ["Método", f"Estimativa ({ano_query})"]
colors_header = ["#185FA5", "#185FA5"]
colors_rows   = [["#E6F1FB", "#E6F1FB"], ["white", "white"], ["#E6F1FB", "#E6F1FB"]]

table = ax.table(
    cellText=estimativas,
    colLabels=columns,
    cellLoc="center",
    loc="center",
    cellColours=colors_rows,
    colColours=colors_header,
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#B5D4F4")
    if row == 0:
        cell.set_text_props(color="white", fontweight="bold")

ax.set_title(f"Estimativas de expectativa de vida para {ano_query}",
             fontsize=12, fontweight="bold", pad=14, color="#0C447C")
plt.tight_layout()
plt.savefig(f"output/tabela_estimativa_{ano_query}_expectativa_de_vida.png", dpi=150, bbox_inches="tight")
plt.show()


###### Q2

# a)
parametros_pareto = [10, 1]

def f(x, parametros):
    return parametros[0] * parametros[1]**parametros[0] / x**(parametros[0] + 1)

def avaliar_funcao(func, params, n, a, b):
    x = np.random.uniform(a, b, n)
    y = func(x, params)
    return x, y

def criar_grid(n, a, b, space="line"):
    if space == "line":
        x = np.linspace(a, b, n)
    elif space == "log":
        x = np.logspace(np.log10(a), np.log10(b), n)
    else:
        print("erro, espaço nao definido")
    return x

def avaliar_funcao_pontos(func, params, x):
    return func(x, params)

def plot_funcao(func, x, y, params, save_path=None):
    x_dense = np.linspace(x.min(), x.max(), 1000)
    y_dense = func(x_dense, params)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, color="tomato", s=5, zorder=5, label="amostras")
    ax.plot(x_dense, y_dense, color="steelblue", zorder=10, lw=2, linestyle="--", label="f(x)")
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

def interpolar(x, y, metodo="spline"):
    if metodo == "linear":
        func = lambda x_new: np.interp(x_new, x, y)
    elif metodo == "spline":
        func = CubicSpline(x, y)
    elif metodo == "akima":
        func = Akima1DInterpolator(x, y)
    else:
        raise ValueError(f"Método '{metodo}' inválido. Use 'linear', 'spline' ou 'akima'.")
    return func

def erro_quadratico(interp_func, x_eval, y_real):
    y_pred = interp_func(x_eval)
    return np.mean((y_pred - y_real) ** 2)

def tabela_mse(metodos_dict, x_eval, y_real, numero_pontos, save_path=None):
    valores = [[nome, f"{erro_quadratico(func, x_eval, y_real):.2e}"]
               for nome, func in metodos_dict.items()]
    n = len(valores)
    colors_rows = [["#E6F1FB", "#E6F1FB"] if i % 2 == 0 else ["white", "white"]
                   for i in range(n)]
    fig, ax = plt.subplots(figsize=(6, 1 + n * 0.6))
    ax.axis("off")
    table = ax.table(
        cellText=valores,
        colLabels=["Método", "MSE"],
        cellLoc="center",
        loc="center",
        cellColours=colors_rows,
        colColours=["#185FA5", "#185FA5"],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#B5D4F4")
        if row == 0:
            cell.set_text_props(color="white", fontweight="bold")
    ax.set_title(f"Erro quadrático médio por método com {numero_pontos} pontos",
                 fontsize=12, fontweight="bold", pad=14, color="#0C447C")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

def pipeline_de_avaliaçao(n, a, b, space, func, param, x_real, y_real, func_name="func"):
    grid = criar_grid(n, a, b, space)
    y    = avaliar_funcao_pontos(func, param, grid)

    intep_lin    = interpolar(grid, y, "linear")
    intep_spline = interpolar(grid, y, "spline")
    intep_akima  = interpolar(grid, y, "akima")

    metodos = {"linear": intep_lin, "spline": intep_spline, "akima": intep_akima}
    save_path = f"output/tabela_{n}_{space}_{func_name}.png"
    tabela_mse(metodos, x_real, y_real, n, save_path=save_path)

    return {
        "linear": erro_quadratico(intep_lin,    x_real, y_real),
        "spline": erro_quadratico(intep_spline, x_real, y_real),
        "akima":  erro_quadratico(intep_akima,  x_real, y_real),
    }


x_amostras, y_amostras = avaliar_funcao(f, parametros_pareto, n=2500, a=1, b=5)

print("x:", x_amostras.round(3))
print("y:", y_amostras.round(3))

plot_funcao(f, x_amostras, y_amostras, parametros_pareto,
            save_path="output/plot_function_f.png")

# b)
grid10 = criar_grid(10, 1, 5, "line")
y_10   = avaliar_funcao_pontos(f, parametros_pareto, grid10)

# c)
intep_lin_10    = interpolar(grid10, y_10, "linear")
intep_spline_10 = interpolar(grid10, y_10, "spline")
intep_akima_10  = interpolar(grid10, y_10, "akima")

mse_linear10 = erro_quadratico(intep_lin_10,    x_amostras, y_amostras)
mse_spline10 = erro_quadratico(intep_spline_10, x_amostras, y_amostras)
mse_akima10  = erro_quadratico(intep_akima_10,  x_amostras, y_amostras)

print(f"MSE linear : {mse_linear10:.2e}")
print(f"MSE spline : {mse_spline10:.2e}")
print(f"MSE akima  : {mse_akima10:.2e}")

metodos_10 = {"linear": intep_lin_10, "spline": intep_spline_10, "akima": intep_akima_10}
tabela_mse(metodos_10, x_amostras, y_amostras, 10,
           save_path="output/tabela_10_line_f.png")

# d) e e)
ns = [10, 15, 20, 30, 50]

resultados_line = {n: pipeline_de_avaliaçao(n, 1, 5, "line", f, parametros_pareto, x_amostras, y_amostras, func_name="f") for n in ns}
resultados_log  = {n: pipeline_de_avaliaçao(n, 1, 5, "log",  f, parametros_pareto, x_amostras, y_amostras, func_name="f") for n in ns}

metodos_nomes = ["linear", "spline", "akima"]
cores = {"line": "steelblue", "log": "tomato"}

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
for ax, metodo in zip(axes, metodos_nomes):
    mse_line = [resultados_line[n][metodo] for n in ns]
    mse_log  = [resultados_log[n][metodo]  for n in ns]
    ax.plot(ns, mse_line, marker="o", color=cores["line"], lw=2, label="linspace")
    ax.plot(ns, mse_log,  marker="o", color=cores["log"],  lw=2, label="logspace")
    ax.set_title(metodo, fontsize=12, fontweight="bold", color="#0C447C")
    ax.set_xlabel("número de pontos no grid")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_yscale("log")

plt.suptitle("f(x) — Linspace vs Logspace por método", fontsize=13,
             fontweight="bold", color="#0C447C", y=1.02)
plt.tight_layout()
plt.savefig("output/plot_comparativo_line_log_f.png", dpi=150, bbox_inches="tight")
plt.show()


###### Q3

def g(x, param):
    return 1 / (1 + np.exp(-param[0] * np.log(x / param[1])))

param_exp = [5, 500]

x_amostras_exp, y_amostras_exp = avaliar_funcao(g, param_exp, n=2500, a=200, b=1000)

print("x:", x_amostras_exp.round(3))
print("y:", y_amostras_exp.round(3))

plot_funcao(g, x_amostras_exp, y_amostras_exp, param_exp,
            save_path="output/plot_function_g.png")

resultados_line_exp = {n: pipeline_de_avaliaçao(n, 200, 1000, "line", g, param_exp, x_amostras_exp, y_amostras_exp, func_name="g") for n in ns}
resultados_log_exp  = {n: pipeline_de_avaliaçao(n, 200, 1000, "log",  g, param_exp, x_amostras_exp, y_amostras_exp, func_name="g") for n in ns}

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
for ax, metodo in zip(axes, metodos_nomes):
    mse_line = [resultados_line_exp[n][metodo] for n in ns]
    mse_log  = [resultados_log_exp[n][metodo]  for n in ns]
    ax.plot(ns, mse_line, marker="o", color=cores["line"], lw=2, label="linspace")
    ax.plot(ns, mse_log,  marker="o", color=cores["log"],  lw=2, label="logspace")
    ax.set_title(metodo, fontsize=12, fontweight="bold", color="#0C447C")
    ax.set_xlabel("número de pontos no grid")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_yscale("log")

plt.suptitle("g(x) — Linspace vs Logspace por método", fontsize=13,
             fontweight="bold", color="#0C447C", y=1.02)
plt.tight_layout()
plt.savefig("output/plot_comparativo_line_log_g.png", dpi=150, bbox_inches="tight")
plt.show()


###### Q4

def h(x):
    return x**3 + 2*x + 5

x_plot = np.linspace(-5, 5, 400)
y_plot = h(x_plot)

plt.plot(x_plot, y_plot, label='$h(x) = x^3 + 2x + 5$', color='blue')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.title('Gráfico da função $h(x) = x^3 + 2x + 5$')
plt.xlabel('x')
plt.ylabel('h(x)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig("output/plot_function_h.png", dpi=150, bbox_inches="tight")
plt.show()

# a derivada é 3x^2+2>0 para todo x então ela é crescente monotonica e
# existe h(a)>0 e h(b)<0 então tem uma única raiz entre -4 e 4

def bissecao(func, a, b, tol=1e-6, max_iter=1000):
    fa = func(a)
    fb = func(b)
    if fa * fb > 0:
        raise ValueError("f(a) e f(b) devem ter sinais opostos.")
    for i in range(max_iter):
        c  = (a + b) / 2
        fc = func(c)
        if abs(fc) < tol or (b - a) / 2 < tol:
            print(f"Bissecção convergiu em {i+1} iterações")
            return c, i + 1
        if fa * fc < 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    raise ValueError(f"Não convergiu em {max_iter} iterações.")

def secante(func, x0, x1, tol=1e-6, max_iter=1000):
    for i in range(max_iter):
        f0 = func(x0)
        f1 = func(x1)
        if abs(f1 - f0) < 1e-12:
            raise ValueError("Denominador muito pequeno, método falhou.")
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        if abs(x2 - x1) < tol:
            print(f"Secante convergiu em {i+1} iterações")
            return x2, i + 1
        x0, x1 = x1, x2
    raise ValueError(f"Secante não convergiu em {max_iter} iterações.")

def newton(func, derive, x0, tol=1e-6, max_iter=1000):
    for i in range(max_iter):
        f0 = func(x0)
        df = derive(x0)
        if abs(df) < 1e-12:
            raise ValueError("Derivada muito pequena, método falhou.")
        x1 = x0 - f0 / df
        if abs(x1 - x0) < tol:
            print(f"Newton convergiu em {i+1} iterações")
            return x1, i + 1
        x0 = x1
    raise ValueError(f"Newton não convergiu em {max_iter} iterações.")

def dh(x):
    return 3*x**2 + 2

raiz_bic, i_bic = bissecao(h, -4, 4, tol=1e-8)
raiz_sec, i_sec = secante(h, -4, 4,  tol=1e-8)
raiz_new, i_new = newton(h, dh, -4,  tol=1e-8)

estimativas = [
    ["Bissecção", f"{raiz_bic:.8f}", str(i_bic)],
    ["Secante",   f"{raiz_sec:.8f}", str(i_sec)],
    ["Newton",    f"{raiz_new:.8f}", str(i_new)],
]

fig, ax = plt.subplots(figsize=(7, 2.5))
ax.axis("off")

columns       = ["Método", "Raiz", "Iterações"]
colors_header = ["#185FA5"] * 3
colors_rows   = [["#E6F1FB"] * 3, ["white"] * 3, ["#E6F1FB"] * 3]

table = ax.table(
    cellText=estimativas,
    colLabels=columns,
    cellLoc="center",
    loc="center",
    cellColours=colors_rows,
    colColours=colors_header,
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor("#B5D4F4")
    if row == 0:
        cell.set_text_props(color="white", fontweight="bold")

ax.set_title("Comparação dos métodos — h(x) = x³ + 2x + 5",
             fontsize=12, fontweight="bold", pad=14, color="#0C447C")
plt.tight_layout()
plt.savefig("output/tabela_raizes_h.png", dpi=150, bbox_inches="tight")
plt.show()