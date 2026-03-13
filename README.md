# CME_pset2
# pset2 — Métodos Numéricos

Implementação de interpolação e busca de raízes para o Problem Set 2.

---

## Conteúdo

- **Q1** — Interpolação de dados de expectativa de vida (linear, regressão linear, cubic spline)
- **Q2** — Interpolação da distribuição de Pareto com grids linspace e logspace
- **Q3** — Interpolação da função logística $g(x)$ com grids linspace e logspace
- **Q4** — Busca de raízes de $h(x) = x^3 + 2x + 5$ via bissecção, secante e Newton

Todos os gráficos e tabelas são salvos automaticamente na pasta `output/`.

---

## Requisitos

- Python 3.11+
- Conda

---

## Instalação

**1. Criar e ativar o ambiente:**

```bash
conda create -n pset2 python=3.11
conda activate pset2
```

**2. Instalar as dependências:**

```bash
pip install -r requirements.txt
```

**3. Rodar o código:**

```bash
python pset2.py
```

---

## Estrutura

```
pset2/
├── pset2.py            # código principal
├── pset2_comp.pdf           # documento LaTeX com os resultados
├── requirements.txt    # dependências
└── output/             # figuras geradas automaticamente
```

---

## Dependências

| Pacote       | Uso                               |
|--------------|-----------------------------------|
| `numpy`      | operações numéricas e grids       |
| `pandas`     | estrutura de dados                |
| `matplotlib` | geração de gráficos e tabelas     |
| `scipy`      | cubic spline e interpolação akima |
