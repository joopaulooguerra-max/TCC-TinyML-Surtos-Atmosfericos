"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  TCC — Classificação de Sinais de Surtos Atmosféricos com TinyML            ║
║  Arquivo : tcc_simulacao_v2.py          (versão corrigida)                  ║
║  O que faz: simula uma linha elétrica de 13,8 kV e gera 3 cenários:         ║
║             • Operação normal (senoidal de 60 Hz com ruído)                 ║
║             • Surto leve (pico 1,5× a 3× o nominal)                        ║
║             • Surto severo (pico acima de 3× o nominal)                     ║
║  Correção:  parâmetros alfa e beta do impulso ajustados conforme             ║
║             Heidler (1985) para atender à norma IEC 60060-1                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

O QUE MUDOU EM RELAÇÃO À VERSÃO ANTERIOR:
  • alfa: de 1/50e-6 (20.000) → para 14.286 s⁻¹
  • beta: de 1/1.2e-6 (833.333) → para 2,16×10⁶ s⁻¹
  Resultado: frente de 1,36 µs e cauda de 51,3 µs — dentro da IEC 60060-1

O QUE É O CIRCUITO:
  Circuito RLC série equivalente simplificado de uma linha de 13,8 kV / 10 km.
  Não é uma linha PI completa — é um modelo simplificado adequado para
  gerar dados de treinamento do classificador.

Dependências: numpy, scipy, matplotlib (já presentes no Google Colab)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

# ══════════════════════════════════════════════════════════════════════════════
# PARÂMETROS DO CIRCUITO
# ══════════════════════════════════════════════════════════════════════════════

# Linha de distribuição 13,8 kV / 10 km (Stevenson, "Análise de Sistemas de Potência")
R_LINHA = 0.01273 * 10    # Ω  — resistência série total
L_LINHA = 0.9337e-3 * 10  # H  — indutância série total
C_LINHA = 12.74e-9 * 10   # F  — capacitância shunt total (referência)
R_ACOPL = 400             # Ω  — resistência de acoplamento (IEC 60060-2)
R_CARGA = 50              # Ω  — resistência equivalente da carga
L_CARGA = 0.1             # H  — indutância equivalente da carga

L_TOT   = L_LINHA + L_CARGA
R_TOT   = R_LINHA + R_CARGA + R_ACOPL

# Fonte AC
VPICO_AC  = 13800 * np.sqrt(2)   # V — 13,8 kV RMS convertido para pico
FREQ_REDE = 60                    # Hz — frequência da rede brasileira

# ──────────────────────────────────────────────────────────────────────────────
# PARÂMETROS DO IMPULSO 1,2/50 µs — CORRIGIDOS (Heidler, 1985)
#
# Explicação simples:
#   alfa controla a descida (cauda) da onda — quanto maior, mais rápido cai
#   beta controla a subida (frente) da onda — quanto maior, mais rápido sobe
#   Os valores abaixo foram calculados para que a onda resultante
#   tenha frente de ~1,2 µs e cauda de ~50 µs, conforme a norma IEC 60060-1
# ──────────────────────────────────────────────────────────────────────────────
ALFA = 14286.0   # s⁻¹ — controla a cauda de 50 µs  ← CORRIGIDO
BETA = 2.16e6    # s⁻¹ — controla a frente de 1,2 µs ← CORRIGIDO

# Discretização do tempo
DT      = 1e-7          # 100 ns — passo de simulação
T_TOTAL = 200e-6        # 200 µs — janela de observação
t       = np.arange(0, T_TOTAL, DT)
N       = len(t)

print("=" * 58)
print("  TCC — Simulação de Surtos Atmosféricos v2 (corrigida)")
print("=" * 58)
print(f"\n  Circuito: R={R_TOT:.1f} Ω | L={L_TOT:.4f} H | C={C_LINHA:.2e} F")
print(f"  Impulso:  alfa={ALFA:.0f} s⁻¹ | beta={BETA:.2e} s⁻¹")
print(f"  Tempo:    passo={DT*1e9:.0f} ns | janela={T_TOTAL*1e6:.0f} µs\n")


# ══════════════════════════════════════════════════════════════════════════════
# FUNÇÕES DE GERAÇÃO DE SINAL
# ══════════════════════════════════════════════════════════════════════════════

def gerar_impulso(vpico_V):
    """
    Gera o impulso atmosférico padrão 1,2/50 µs (norma IEC 60060-1).

    Como funciona (explicação simples):
      A onda de um raio sobe muito rápido (em microsegundos)
      e depois cai devagar. Matematicamente isso é modelado como
      a diferença de duas exponenciais — uma que sobe rápido (beta)
      e outra que cai devagar (alfa).

    Parâmetro:
      vpico_V: tensão de pico desejada em Volts
    """
    onda = vpico_V * (np.exp(-ALFA * t) - np.exp(-BETA * t))
    onda = onda / np.max(onda) * vpico_V  # normaliza para o pico exato
    onda[onda < 0] = 0                    # garante que não fica negativo
    return onda


def simular_circuito(v_entrada):
    """
    Resolve as equações do circuito RLC pelo método de Euler.

    Como funciona (explicação simples):
      É como calcular passo a passo o que acontece com a corrente
      e a tensão a cada 100 nanosegundos. Em cada instante,
      calculamos: dada a tensão de entrada agora, como a corrente
      vai mudar no próximo instante?

    Retorna a tensão medida na resistência de carga.
    """
    i   = np.zeros(N)    # corrente inicial zero
    v_c = np.zeros(N)    # tensão no capacitor inicial zero

    for n in range(1, N):
        # Quanto a corrente muda neste instante
        di   = (v_entrada[n-1] - R_TOT * i[n-1] - v_c[n-1]) / L_TOT
        i[n] = i[n-1] + di * DT
        # Quanto a tensão no capacitor muda
        dvc     = i[n] / C_LINHA
        v_c[n]  = v_c[n-1] + dvc * DT

    return R_CARGA * i   # tensão de saída na carga


# ══════════════════════════════════════════════════════════════════════════════
# GERAÇÃO DOS 3 CENÁRIOS
# ══════════════════════════════════════════════════════════════════════════════

np.random.seed(42)

# ── Cenário 0: Normal ─────────────────────────────────────────────────────────
# O que é: rede elétrica funcionando normalmente — senoidal de 60 Hz com ruído
print("[0] Simulando cenário NORMAL...")
v_ac     = VPICO_AC * np.sin(2 * np.pi * FREQ_REDE * t)
v_ruido  = np.random.normal(0, VPICO_AC * 0.01, N)
v_normal = simular_circuito(v_ac + v_ruido)
print(f"    RMS na carga: {np.sqrt(np.mean(v_normal**2)):.2f} V")

# ── Cenário 1: Surto Leve ─────────────────────────────────────────────────────
# O que é: pico entre 1,5× e 3× a tensão nominal
# Representa: raio caiu longe da linha (indução atmosférica indireta)
print("\n[1] Simulando cenário SURTO LEVE...")
VPICO_LEVE  = VPICO_AC * 2.0   # 2× o nominal (dentro da faixa leve)
v_imp_leve  = gerar_impulso(VPICO_LEVE)
v_surto_leve = simular_circuito(v_ac + v_imp_leve)
print(f"    Pico do impulso: {v_imp_leve.max()/1000:.2f} kV")
print(f"    Pico na carga:   {v_surto_leve.max():.2f} V")

# ── Cenário 2: Surto Severo ───────────────────────────────────────────────────
# O que é: pico acima de 3× a tensão nominal
# Representa: raio caiu diretamente na linha ou muito perto
print("\n[2] Simulando cenário SURTO SEVERO...")
VPICO_SEVERO  = VPICO_AC * 5.0   # 5× o nominal (surto severo)
v_imp_severo  = gerar_impulso(VPICO_SEVERO)
v_surto_severo = simular_circuito(v_ac + v_imp_severo)
print(f"    Pico do impulso: {v_imp_severo.max()/1000:.2f} kV")
print(f"    Pico na carga:   {v_surto_severo.max():.2f} V")


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICAÇÃO DO IMPULSO — confirma que está dentro da norma IEC 60060-1
# ══════════════════════════════════════════════════════════════════════════════

print("\n[3] Verificando parâmetros do impulso...")
impulso_ref = gerar_impulso(1.0)
idx_pico = np.argmax(impulso_ref)
Vp = impulso_ref[idx_pico]
idx_30   = np.argmax(impulso_ref >= 0.30 * Vp)
idx_90   = np.argmax(impulso_ref >= 0.90 * Vp)
T1_real  = (t[idx_90] - t[idx_30]) / 0.6 * 1e6
idx_caud = idx_pico + np.argmax(impulso_ref[idx_pico:] <= 0.50 * Vp)
T2_real  = t[idx_caud] * 1e6

print(f"    Frente (T1): {T1_real:.2f} µs  (norma: 1,2 µs ± 30% → 0,84 a 1,56)")
print(f"    Cauda  (T2): {T2_real:.2f} µs  (norma: 50 µs ± 20% → 40 a 60)")
ok1 = 0.84 <= T1_real <= 1.56
ok2 = 40   <= T2_real <= 60
print(f"    Resultado: {'✓ APROVADO — dentro da IEC 60060-1' if ok1 and ok2 else '✗ REPROVADO'}")


# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICOS — equivalente ao Scope do Simulink
# ══════════════════════════════════════════════════════════════════════════════

t_us = t * 1e6   # converte para microssegundos

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle(
    'TCC — Simulação de Surtos Atmosféricos v2\n'
    'Linha de Distribuição 13,8 kV — Circuito RLC Equivalente Simplificado',
    fontsize=13
)

cenarios = [
    ('Normal',        v_ac + v_ruido,    v_normal,       'steelblue'),
    ('Surto Leve',    v_ac + v_imp_leve, v_surto_leve,   'darkorange'),
    ('Surto Severo',  v_ac + v_imp_severo, v_surto_severo, 'crimson'),
]

for col, (nome, v_ent, v_sai, cor) in enumerate(cenarios):
    axes[0, col].plot(t_us, v_ent / 1000, color=cor, lw=0.8)
    axes[0, col].set_title(f'{nome} — Entrada (kV)')
    axes[0, col].set_ylabel('Tensão (kV)')
    axes[0, col].grid(True, alpha=0.3)

    axes[1, col].plot(t_us, v_sai, color=cor, lw=0.8)
    axes[1, col].set_title(f'{nome} — Saída na Carga (V)')
    axes[1, col].set_ylabel('Tensão (V)')
    axes[1, col].set_xlabel('Tempo (µs)')
    axes[1, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('simulacao_v2.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n[4] Gráfico salvo em 'simulacao_v2.png'")


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTAÇÃO DOS DADOS
# ══════════════════════════════════════════════════════════════════════════════

np.savez('dados_simulacao_v2.npz',
    tempo            = t,
    tensao_normal    = v_normal,
    tensao_surto_leve    = v_surto_leve,
    tensao_surto_severo  = v_surto_severo,
    v_impulso_leve   = v_imp_leve,
    v_impulso_severo = v_imp_severo,
    alfa = np.array([ALFA]),
    beta = np.array([BETA]),
)

savemat('dados_simulacao_v2.mat', {
    'tempo': t, 'tensao_normal': v_normal,
    'tensao_surto_leve': v_surto_leve,
    'tensao_surto_severo': v_surto_severo,
})

print("[5] Dados exportados:")
print("    ├── dados_simulacao_v2.npz  → uso no Colab (próximo passo)")
print("    └── dados_simulacao_v2.mat  → compatível com MATLAB")
print("\n✓ Passo 1 concluído! Rode agora o tcc_augmentation_3classes.py")
