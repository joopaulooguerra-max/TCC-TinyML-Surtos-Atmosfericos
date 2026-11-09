# Robustez em 10 sementes

Média ± desvio padrão (mínimo a máximo) da acurácia, em 300 sinais novos por cenário.

| Cenário | Random Forest | Rede INT8 |
|---|:-:|:-:|
| A Controle | 98.0 ± 0.1  (97.7-98.0) | 97.9 ± 0.4  (97.3-98.3) |
| B Ruído maior | 96.9 ± 1.1  (94.3-98.0) | 98.0 ± 0.3  (97.7-98.3) |
| C Amplitude fora | 97.1 ± 0.3  (96.7-97.7) | 97.7 ± 0.2  (97.3-98.0) |
| D Impulso nas bordas | 94.8 ± 0.4  (94.0-95.7) | 94.4 ± 1.4  (92.3-97.0) |
| E Forma diferente | 96.4 ± 0.3  (96.0-97.0) | 96.9 ± 0.2  (96.7-97.3) |
| F Harmônicos | 98.6 ± 0.2  (98.3-98.7) | 98.4 ± 0.4  (97.7-98.7) |
| G B+C+D+E | 96.6 ± 0.5  (95.3-97.0) | 95.9 ± 0.8  (94.7-97.3) |