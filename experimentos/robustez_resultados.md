# Teste de robustez

Treino com 900 sinais do dataset; teste com 300 sinais novos por cenário (100 por classe).

| Cenário | Random Forest | INT8 |
|---|:-:|:-:|
| A  Controle (mesma distribuição) | 98.0% | 98.0% |
| B  Ruído maior (0,10 a 0,20) | 97.0% | 98.7% |
| C  Amplitude da senoide fora da faixa | 97.0% | 98.0% |
| D  Impulso nas bordas da janela | 95.3% | 95.3% |
| E  Forma do impulso diferente (0,6x a 1,4x) | 97.0% | 97.7% |
| F  Harmônicos de 3a e 5a ordem | 98.7% | 98.0% |
| G  B + C + D + E juntos | 97.3% | 96.7% |