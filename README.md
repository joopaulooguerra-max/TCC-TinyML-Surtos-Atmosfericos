#  TCC — Sistema Embarcado com TinyML para Classificação de Surtos Atmosféricos

> **Trabalho de Conclusão de Curso — Engenharia**  
> Tema: *Sistema Embarcado Baseado em TinyML para Classificação de Sinais de Surtos Atmosféricos em Redes de Distribuição Elétrica*

---

##  Sobre o projeto

Este trabalho propõe o desenvolvimento de um sistema embarcado de aprendizado de máquina capaz de **classificar sinais elétricos** em redes de distribuição de média tensão (13,8 kV), distinguindo automaticamente entre operação normal e dois níveis de severidade de surto atmosférico.

O modelo de classificação é treinado em ambiente Python (Google Colab) e exportado para o formato **TensorFlow Lite (INT8)**, viabilizando sua execução diretamente no microcontrolador **ESP32-S3** sem dependência de conectividade externa — paradigma conhecido como **TinyML**.

### Por que esse tema é relevante?

O Brasil possui uma das maiores incidências de raios do mundo, com cerca de **77,8 milhões de descargas atmosféricas por ano** (ELAT/INPE). Surtos atmosféricos são uma das principais causas de falhas em equipamentos de redes de distribuição, causando danos a transformadores, religadores e capacitores. Um dispositivo embarcado capaz de **registrar e classificar automaticamente** a severidade do evento — sem necessidade de equipe de campo para diagnóstico — representa um avanço prático para concessionárias de energia.

---

##  Estrutura do repositório

```
TCC-TinyML-Surtos-Atmosfericos/
│
├── simulacao/
│   ├── tcc_simulacao_v2.py              # Simula o circuito RLC e gera os 3 cenários
│   └── tcc_augmentation_3classes.py     # Gera 1500 exemplos — 500 por classe
│
├── modelo/
│   ├── tcc_pipeline_3classes_v2.py      # Treina e valida o classificador Random Forest
│   └── normalizacao_3classes_v2.npz     # Parâmetros min/max para o firmware ESP32-S3
│
├── dados/
│   └── dados_augmentados_3classes.npz   # Dataset completo — 3 classes balanceadas
│
├── resultados/
│   └── resultados_v2.png                # Matriz de confusão e importância das features
│
└── README.md
```

---

##  As 3 classes do classificador

| Classe | Nome | Descrição | Amplitude do pico |
|:------:|------|-----------|:-----------------:|
| **0** | Normal | Rede operando normalmente — senoidal 60 Hz com ruído | Nominal |
| **1** | Surto Leve | Indução atmosférica indireta — raio caiu longe da linha | 1,5× a 3× |
| **2** | Surto Severo | Descarga atmosférica direta ou muito próxima da linha | Acima de 3× |

---

##  Pipeline do projeto

```
Simulação RLC          Data Augmentation       Extração de Features
(tcc_simulacao_v2)  →  (1500 exemplos)      →  (8 características)
                                                        ↓
                                              Treinamento Random Forest
                                                        ↓
                                              Validação K-Fold (k=5)
                                                        ↓
                                              Exportação TFLite INT8
                                                        ↓
                                              Firmware ESP32-S3 (model.h)
```

---

##  Base de dados

Os dados são **sintéticos**, gerados por equações físicas fundamentadas em normas e literatura técnica. Cada sinal é composto por três componentes somados:

```
Sinal = Senoidal 60 Hz  +  Ruído gaussiano  +  Impulso atmosférico (quando há surto)
```

### Parâmetros do impulso atmosférico — norma IEC 60060-1

O impulso padrão 1,2/50 µs é modelado pela dupla exponencial:

```
v(t) = Vp × (e^(-α·t) − e^(-β·t))
```

| Parâmetro | Valor | Significado |
|-----------|-------|-------------|
| α (alfa) | 14.286 s⁻¹ | Controla a cauda de 50 µs |
| β (beta) | 2,16 × 10⁶ s⁻¹ | Controla a frente de 1,2 µs |
| T1 (frente) | 1,36 µs | Dentro da tolerância IEC ±30% |
| T2 (cauda) | 51,3 µs | Dentro da tolerância IEC ±20% |

> Valores baseados em **Heidler (1985)** e validados conforme **IEC 60060-1**.

### Parâmetros do circuito RLC equivalente simplificado

| Parâmetro | Valor | Referência |
|-----------|-------|------------|
| Tensão nominal | 13,8 kV / 60 Hz | PRODIST ANEEL |
| Comprimento da linha | 10 km | Alimentador urbano típico |
| Resistência série (R) | 0,1273 Ω | Stevenson (2006) |
| Indutância série (L) | 9,337 mH | Stevenson (2006) |
| Capacitância shunt (C) | 127,4 nF | Stevenson (2006) |

---

##  Características extraídas (features)

Cada sinal de 2001 amostras é resumido em **8 números** que descrevem seu comportamento nos domínios do tempo e da frequência:

| # | Feature | Domínio | O que mede |
|---|---------|---------|------------|
| 1 | RMS | Tempo | Energia média do sinal |
| 2 | Pico | Tempo | Valor absoluto máximo |
| 3 | Fator de Crista | Tempo | Razão Pico/RMS — explode em surtos |
| 4 | Desvio Padrão | Tempo | Dispersão dos valores |
| 5 | Prop_HF_100kHz | Frequência | Fração da energia acima de 100 kHz |
| 6 | Centroide Espectral | Frequência | Frequência média ponderada pela energia |
| 7 | Freq_Dom_10kHz | Frequência | Frequência dominante acima de 10 kHz |
| 8 | Entropia Espectral | Frequência | Dispersão do espectro |

> **Nota técnica:** a feature `Mag_60Hz` foi removida por ser inválida para a janela de 200 µs utilizada — a resolução da FFT de 5 kHz/bin impede a identificação da componente de 60 Hz. Foi substituída pelo Centroide Espectral, matematicamente adequado para esta configuração.

---

##  Resultados

### Relatório de classificação

| Classe | Precisão | Recall | F1-Score |
|--------|:--------:|:------:|:--------:|
| Normal | 1,00 | 1,00 | 1,00 |
| Surto Leve | 0,97 | 0,96 | 0,96 |
| Surto Severo | 0,96 | 0,97 | 0,97 |
| **Média** | **0,98** | **0,98** | **0,98** |

### Validação cruzada K-Fold (k=5)

| Fold | Acurácia |
|:----:|:--------:|
| 1 | 98,3% |
| 2 | 97,3% |
| 3 | 98,7% |
| 4 | 97,0% |
| 5 | 97,7% |
| **Média** | **97,8% ± 0,62%** |

> A baixa variância entre os folds confirma que o modelo é **consistente** e não depende de um split específico para apresentar bom desempenho.

### Importância das features

| Feature | Importância |
|---------|:-----------:|
| Desvio Padrão | 37,7% |
| Pico | 24,0% |
| Fator de Crista | 16,0% |
| RMS | 9,1% |
| Prop_HF_100kHz | 5,9% |
| Entropia | 3,1% |
| Centroide | 2,7% |
| Freq_Dom_10kHz | 1,5% |

---

##  Como executar

### Pré-requisitos
- Google Colab ou Python 3.10+
- Bibliotecas: `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `tensorflow`

### Ordem de execução

```bash
# 1. Gera os sinais simulados e exporta dados_simulacao_v2.npz
python simulacao/tcc_simulacao_v2.py

# 2. Gera o dataset de 1500 exemplos e exporta dados_augmentados_3classes.npz
python simulacao/tcc_augmentation_3classes.py

# 3. Treina o classificador, valida e salva o modelo
python modelo/tcc_pipeline_3classes_v2.py
```

---

##  Tecnologias utilizadas

| Tecnologia | Uso |
|------------|-----|
| Python 3.10+ | Linguagem principal |
| NumPy / SciPy | Simulação e processamento de sinais |
| Scikit-Learn | Random Forest e validação |
| TensorFlow Lite | Exportação do modelo para microcontrolador |
| ESP32-S3 | Microcontrolador alvo (TinyML) |
| Google Colab | Ambiente de desenvolvimento |

---

##  Referências técnicas

- **IEC 60060-1** — High-voltage test techniques: General definitions and test requirements
- **IEC 60060-2** — High-voltage test techniques: Measuring systems
- **ANEEL — PRODIST Módulo 2 e 8** — Procedimentos de Distribuição de Energia Elétrica
- **HEIDLER, F.** (1985). Travelling current source model for LEMP calculation. *6th Int. Zurich Symp. on EMC*
- **STEVENSON, W. D.** (2006). *Elementos de Análise de Sistemas de Potência*. McGraw-Hill
- **WARDEN, P.; SITUNAYAKE, D.** (2020). *TinyML*. O'Reilly Media
- **BHATT, U. et al.** (2016). Power quality disturbance classification. *IEEE Trans. Power Delivery*
- **ELAT/INPE** — Grupo de Eletricidade Atmosférica, Instituto Nacional de Pesquisas Espaciais

---

##  Autor

**João Paulo Guerra**  
Curso de Engenharia — TCC 2026  
Orientador: Prof(a). João Ferreira

---

##  Observações importantes

- Os dados utilizados são **sintéticos**, gerados por equações físicas fundamentadas em normas técnicas internacionais. Não representam medições reais de campo em redes elétricas.
- Os resultados de 98% de acurácia foram obtidos em dados sintéticos de teste. O desempenho em sinais reais de campo pode diferir e constitui trabalho futuro.
- O circuito utilizado é um **modelo RLC equivalente simplificado**, não uma linha PI completa — adequado para fins de geração de dados de treinamento.
