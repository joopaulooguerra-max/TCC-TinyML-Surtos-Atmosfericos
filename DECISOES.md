# Registro de decisões

## Decisões registradas

### D-001 — Manter dados sintéticos como etapa inicial
- **Estado:** adotada.
- **Motivo:** permite construir e testar o pipeline antes de obter sinais reais.
- **Condição:** conclusões limitadas aos cenários simulados; a falta de validação real é declarada.

### D-002 — Não tratar a acurácia como prova de generalização
- **Estado:** adotada.
- **Motivo:** a métrica foi calculada em dados sintéticos gerados pelo mesmo modelo que define as classes.
- **Consequência:** apresentar os 97,8% (k-fold) e 98,3% (INT8) como separação dentro do gerador. Teste de robustez previsto.

### D-003 — Usar o ESP32-S3 como plataforma pretendida
- **Estado:** adotada, não validada em hardware.
- **Condição:** confirmar aquisição do sinal, memória, latência e inferência no chip.

### D-004 — Mudar o foco de "mitigação preditiva" para classificação
- **Data:** 2026-11-05
- **Contexto:** o tema anterior prometia prever e mitigar, o que exigiria hardware e dados reais.
- **Decisão:** classificar janelas em normal, surto leve e surto severo, em simulação.
- **Impactos:** resumo, introdução e objetivos reescritos.

### D-005 — Substituir características inválidas ou redundantes
- **Data:** 2026-11-05
- **Contexto:** a magnitude em 60 Hz caía no bin 0 (resolução de 5 kHz) e a energia era RMS² × N.
- **Decisão:** trocar por centroide espectral e desvio padrão, e ajustar os limiares de frequência para 100 kHz e 10 kHz.
- **Impactos:** `tcc_pipeline_3classes_v2.py` e o capítulo de metodologia.

### D-006 — Ajustar o impulso numericamente às tolerâncias da IEC 60060-1
- **Data:** 2026-11-05
- **Contexto:** `alfa` = 1/50 µs e `beta` = 1/1,2 µs davam frente de cerca de 2,7 µs, fora da tolerância.
- **Decisão:** usar `alfa` = 14286 s⁻¹ e `beta` = 2,16×10⁶ s⁻¹, achados por ajuste numérico, e não atribuí-los a uma publicação.
- **Impactos:** `tcc_simulacao_v2.py`, `tcc_augmentation_3classes.py`.

### D-007 — Dataset gerado sem passar pelo circuito RLC
- **Data:** 2026-11-05
- **Contexto:** a simulação RLC e o gerador de dados são independentes.
- **Decisão:** manter o gerador direto (senoide + ruído + impulso) e descrever o circuito como ilustrativo, com R, L e C do MATLAB (735 kV), sem relação com 13,8 kV.
- **Alternativa não adotada:** acoplar o dataset à simulação do circuito. Pode ser trabalho futuro.

### D-008 — Rede neural pequena para a exportação TFLite
- **Data:** 2026-11-05
- **Contexto:** a Random Forest não converte direto para TensorFlow Lite.
- **Decisão:** usar uma rede 8-16-8-3 com INT8. A Random Forest fica como referência (99,0% contra 98,3%).

### D-009 — Quantizar a entrada com arredondamento e saturação
- **Data:** 2026-11-05
- **Contexto:** truncar em vez de arredondar mudou 1 a 2 previsões em 300.
- **Decisão:** o firmware usa `roundf()` e limita o valor entre -128 e 127.

## Modelo para novas decisões

### D-XXX — Título curto
- **Data:** AAAA-MM-DD
- **Estado:** proposta / adotada / rejeitada / substituída.
- **Contexto:** qual problema precisa ser resolvido?
- **Opções consideradas:** quais alternativas foram avaliadas?
- **Decisão:** o que foi escolhido?
- **Justificativa:** por que essa opção foi escolhida?
- **Impactos:** quais arquivos, testes ou resultados precisam mudar?
