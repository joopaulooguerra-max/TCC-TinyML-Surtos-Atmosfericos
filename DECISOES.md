# Registro de decisões

Registre aqui decisões que alterem o escopo, a metodologia, a simulação ou a avaliação. Use o formato abaixo para facilitar o histórico.

## Decisões registradas

### D-001 — Manter dados sintéticos como etapa inicial
- **Estado:** adotada provisoriamente.
- **Motivo:** permite construir e testar o pipeline antes de obter sinais reais.
- **Condição:** as conclusões serão limitadas aos cenários simulados; a falta de validação real será explicitada.

### D-002 — Não tratar 98% de acurácia como prova de generalização
- **Estado:** adotada.
- **Motivo:** a métrica atual foi calculada em dados sintéticos gerados para o mesmo experimento.
- **Consequência:** repetir a avaliação após revisar o gerador e usar cenários completos reservados para teste.

### D-003 — Usar ESP32-S3 como plataforma embarcada pretendida
- **Estado:** objetivo de implementação, ainda não validado.
- **Motivo:** é a plataforma indicada no planejamento atual do TCC.
- **Condição:** confirmar viabilidade de aquisição do sinal, memória, latência e inferência após escolher e treinar o modelo final.

## Modelo para novas decisões

### D-XXX — Título curto
- **Data:** AAAA-MM-DD
- **Estado:** proposta / adotada / rejeitada / substituída.
- **Contexto:** qual problema precisa ser resolvido?
- **Opções consideradas:** quais alternativas foram avaliadas?
- **Decisão:** o que foi escolhido?
- **Justificativa:** por que essa opção foi escolhida?
- **Impactos:** quais arquivos, testes ou resultados precisam mudar?
