import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# =========================================================
# 1. PARÂMETROS DO MODELO
# =========================================================

V0 = 8444961241.04084 / 1_000_000   # Valor dos ativos em R$ milhões
D = 6230511000 / 1_000_000          # Valor da dívida em R$ milhões

sigma_V = 0.236369868498767         # Volatilidade anual dos ativos
r = 0.1075                          # Taxa livre de risco anual
T = 3                               # Horizonte em anos

n_simulations = 1000
steps_per_year = 252
n_steps = int(T * steps_per_year)
dt = T / n_steps

np.random.seed(42)


# =========================================================
# 2. FUNÇÕES DO MODELO
# =========================================================

def calculate_dd(V, D, sigma_V, r, tau):
    tau = np.maximum(tau, 1e-8)

    return (
        np.log(V / D)
        + (r - 0.5 * sigma_V**2) * tau
    ) / (sigma_V * np.sqrt(tau))


def calculate_pd(dd):
    return norm.cdf(-dd)


def simulate_asset_paths(V0, r, sigma_V, T, n_steps, n_simulations):
    dt = T / n_steps
    paths = np.zeros((n_simulations, n_steps + 1))
    paths[:, 0] = V0

    for t in range(1, n_steps + 1):
        z = np.random.standard_normal(n_simulations)

        paths[:, t] = paths[:, t - 1] * np.exp(
            (r - 0.5 * sigma_V**2) * dt
            + sigma_V * np.sqrt(dt) * z
        )

    return paths


# =========================================================
# 3. DD E PD INICIAIS
# =========================================================

DD0 = calculate_dd(V0, D, sigma_V, r, T)
PD0 = calculate_pd(DD0)

print("=" * 60)
print("MODELO ESTRUTURAL DE RISCO DE CRÉDITO")
print("=" * 60)
print(f"Valor inicial dos ativos: R$ {V0:,.2f} milhões")
print(f"Valor da dívida:          R$ {D:,.2f} milhões")
print(f"Volatilidade dos ativos:  {sigma_V:.2%}")
print(f"Taxa livre de risco:      {r:.2%}")
print(f"Prazo:                    {T} anos")
print("-" * 60)
print(f"Distance to Default:      {DD0:.4f}")
print(f"Probability of Default:   {PD0:.4%}")
print("=" * 60)


# =========================================================
# 4. SIMULAÇÃO DAS TRAJETÓRIAS DOS ATIVOS
# =========================================================

asset_paths = simulate_asset_paths(
    V0=V0,
    r=r,
    sigma_V=sigma_V,
    T=T,
    n_steps=n_steps,
    n_simulations=n_simulations
)

time_grid = np.linspace(0, T, n_steps + 1)


# =========================================================
# 5. CÁLCULO DA DD E DA PD AO LONGO DO TEMPO
# =========================================================

remaining_time = np.maximum(T - time_grid, 1e-8)

dd_paths = np.zeros_like(asset_paths)
pd_paths = np.zeros_like(asset_paths)

for t in range(n_steps + 1):
    dd_paths[:, t] = calculate_dd(
        asset_paths[:, t],
        D,
        sigma_V,
        r,
        remaining_time[t]
    )

    pd_paths[:, t] = calculate_pd(dd_paths[:, t])


# =========================================================
# 6. RESULTADOS DA SIMULAÇÃO
# =========================================================

final_assets = asset_paths[:, -1]
default_events = final_assets < D
simulated_PD = default_events.mean()

print(f"PD simulada no vencimento: {simulated_PD:.4%}")
print(f"Valor médio dos ativos no vencimento: R$ {final_assets.mean():,.2f} milhões")
print(f"Mediana dos ativos no vencimento: R$ {np.median(final_assets):,.2f} milhões")


# =========================================================
# 7. GRÁFICO PRINCIPAL — DESIGN PROFISSIONAL
# =========================================================

path_index = 0

selected_asset_path = asset_paths[path_index]
selected_pd_path = pd_paths[path_index]

fig = plt.figure(figsize=(16, 7.5), facecolor="white")

ax1 = fig.add_axes([0.07, 0.17, 0.58, 0.68])
ax2 = ax1.twinx()
ax3 = fig.add_axes([0.73, 0.17, 0.20, 0.68])


# Cores institucionais
asset_color = "#1f4e79"
debt_color = "#8b1a1a"
pd_color = "#2f6b3f"
neutral_color = "#2b2b2b"
grid_color = "#d9d9d9"


# ---------------------------------------------------------
# Painel principal: valor dos ativos
# ---------------------------------------------------------

ax1.plot(
    time_grid,
    selected_asset_path,
    linewidth=2.4,
    color=asset_color,
    label="Asset value"
)

ax1.axhline(
    D,
    linestyle="--",
    linewidth=2.0,
    color=debt_color,
    label="Debt threshold"
)

ax1.fill_between(
    time_grid,
    D,
    selected_asset_path,
    where=selected_asset_path >= D,
    color=asset_color,
    alpha=0.06
)

ax1.fill_between(
    time_grid,
    D,
    selected_asset_path,
    where=selected_asset_path < D,
    color=debt_color,
    alpha=0.12
)

ax1.set_xlabel("Time horizon (years)", fontsize=11)
ax1.set_ylabel("Asset value (R$ million)", fontsize=11, color=asset_color)

ax1.tick_params(axis="y", labelcolor=asset_color, labelsize=10)
ax1.tick_params(axis="x", labelsize=10)

ax1.grid(True, color=grid_color, linewidth=0.8, alpha=0.6)

for spine in ["top", "right"]:
    ax1.spines[spine].set_visible(False)

ax1.spines["left"].set_color("#777777")
ax1.spines["bottom"].set_color("#777777")


# ---------------------------------------------------------
# Segundo eixo: probabilidade de default
# ---------------------------------------------------------

ax2.plot(
    time_grid,
    selected_pd_path,
    linewidth=2.2,
    color=pd_color,
    label="Default probability"
)

ax2.set_ylabel("Default probability", fontsize=11, color=pd_color)
ax2.tick_params(axis="y", labelcolor=pd_color, labelsize=10)

max_pd_axis = max(0.01, min(1, selected_pd_path.max() * 1.35))
ax2.set_ylim(0, max_pd_axis)

for spine in ["top"]:
    ax2.spines[spine].set_visible(False)

ax2.spines["right"].set_color("#777777")


# ---------------------------------------------------------
# Painel lateral: distribuição da DD
# ---------------------------------------------------------

dd_min = min(-4, DD0 - 1.0)
dd_max = max(4, DD0 + 1.0)

dd_range = np.linspace(dd_min, dd_max, 500)
density = norm.pdf(dd_range)

ax3.plot(
    density,
    dd_range,
    linewidth=2.4,
    color=neutral_color,
    label="Standard normal density"
)

ax3.fill_betweenx(
    dd_range,
    0,
    density,
    where=dd_range <= -DD0,
    color=debt_color,
    alpha=0.18,
    label="Default region"
)

ax3.axhline(
    DD0,
    linestyle="--",
    linewidth=2.0,
    color=asset_color,
    label="Initial DD"
)

ax3.scatter(
    norm.pdf(DD0),
    DD0,
    color=asset_color,
    s=55,
    zorder=5
)

ax3.annotate(
    f"DD = {DD0:.2f}\nPD = {PD0:.2%}",
    xy=(norm.pdf(DD0), DD0),
    xytext=(density.max() * 0.42, DD0 + 0.75),
    arrowprops=dict(
        arrowstyle="->",
        linewidth=1.1,
        color="#444444"
    ),
    fontsize=10,
    color="#222222",
    bbox=dict(
        boxstyle="round,pad=0.35",
        fc="white",
        ec="#b0b0b0",
        alpha=0.95
    )
)

ax3.set_xlabel("Density", fontsize=11)
ax3.set_ylabel("Distance to Default", fontsize=11)
ax3.set_title("DD Distribution", fontsize=12, fontweight="bold")

ax3.tick_params(axis="both", labelsize=10)
ax3.grid(True, color=grid_color, linewidth=0.8, alpha=0.6)

for spine in ["top", "right"]:
    ax3.spines[spine].set_visible(False)

ax3.spines["left"].set_color("#777777")
ax3.spines["bottom"].set_color("#777777")


# ---------------------------------------------------------
# Títulos
# ---------------------------------------------------------

fig.suptitle(
    "Structural Credit Risk Model",
    fontsize=18,
    fontweight="bold",
    x=0.07,
    y=0.96,
    ha="left"
)

fig.text(
    0.07,
    0.915,
    (
        f"Asset value simulation, debt threshold and implied default probability "
        f"| V₀ = R$ {V0:,.0f}m | D = R$ {D:,.0f}m | "
        f"σᵥ = {sigma_V:.2%} | r = {r:.2%} | T = {T} years"
    ),
    fontsize=10.5,
    color="#444444",
    ha="left"
)


# ---------------------------------------------------------
# Legenda consolidada
# ---------------------------------------------------------

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines3, labels3 = ax3.get_legend_handles_labels()

fig.legend(
    lines1 + lines2 + lines3,
    labels1 + labels2 + labels3,
    loc="lower center",
    ncol=5,
    frameon=False,
    fontsize=10,
    bbox_to_anchor=(0.5, 0.035)
)


# ---------------------------------------------------------
# Caixa de métricas
# ---------------------------------------------------------

metrics_text = (
    f"Initial DD: {DD0:.2f}\n"
    f"Initial PD: {PD0:.2%}\n"
    f"Simulated PD: {simulated_PD:.2%}"
)

fig.text(
    0.655,
    0.78,
    metrics_text,
    fontsize=10.5,
    color="#222222",
    bbox=dict(
        boxstyle="round,pad=0.45",
        fc="#f7f7f7",
        ec="#c7c7c7",
        alpha=1
    )
)

plt.show()


# =========================================================
# 8. DISTRIBUIÇÃO FINAL DOS ATIVOS
# =========================================================

plt.figure(figsize=(12, 6), facecolor="white")

plt.hist(
    final_assets,
    bins=50,
    density=True,
    alpha=0.7,
    color=asset_color,
    edgecolor="white"
)

plt.axvline(
    D,
    linestyle="--",
    linewidth=2,
    color=debt_color,
    label="Default threshold"
)

plt.title(
    "Simulated Distribution of Asset Value at Maturity",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Asset value at maturity (R$ million)")
plt.ylabel("Density")
plt.legend(frameon=False)
plt.grid(True, color=grid_color, alpha=0.6)

plt.show()
