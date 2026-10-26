# Sistema Embarcado Baseado em TinyML para Classificação de Surtos Atmosféricos em Redes de Distribuição Elétrica

Projeto de Trabalho de Conclusão de Curso (TCC) em Engenharia da Computação. O objetivo de pesquisa é estudar a classificação de sinais elétricos associados à operação normal e a surtos atmosféricos, com foco futuro em execução embarcada usando TinyML e ESP32-S3.

> **Estado atual:** protótipo experimental com sinais sintéticos/simulados. Os resultados atuais não comprovam desempenho em sinais reais nem constituem um sistema de proteção elétrica validado.

## Conteúdo do repositório

- `CONTEXTO_PROJETO.md`: resumo técnico, estado atual e limitações conhecidas.
- `DECISOES.md`: decisões de projeto e pontos ainda em aberto.
- `PROXIMOS_PASSOS.md`: plano de trabalho priorizado.
- `notebooks/ClassificadorTCC.ipynb`: notebook atual de simulação, geração de exemplos e treinamento.
- `simulacao/`: código de simulação a organizar e validar.
- `dados/`: instruções e metadados dos conjuntos de dados; arquivos grandes/gerados não devem ser adicionados sem necessidade.
- `experimentos/`: registro de métricas, configurações e resultados.
- `modelos/`: modelos treinados e artefatos de exportação, quando estiverem prontos.
- `firmware/`: código para a plataforma embarcada, quando existir.

## Fluxo experimental atual

1. Simular uma resposta elétrica simplificada e um impulso transitório.
2. Gerar exemplos sintéticos de três classes: `Normal`, `Surto Leve` e `Surto Severo`.
3. Extrair oito características do sinal.
4. Treinar e avaliar um classificador Random Forest.
5. Em etapas futuras, avaliar um modelo compatível com TinyML, quantização e execução no ESP32-S3.

## Como abrir o notebook

O notebook foi desenvolvido para execução em ambiente Python, incluindo Google Colab. Abra `notebooks/ClassificadorTCC.ipynb` no GitHub e selecione **Open in Colab** se essa opção estiver disponível, ou baixe o arquivo e abra-o no Jupyter.

Dependências indicadas pelo notebook: `numpy`, `scipy`, `matplotlib` e `scikit-learn`. Consulte `requirements.txt`.

## Resultados preliminares registrados

O notebook atual registra um conjunto sintético de 1.500 janelas (500 por classe), 2.001 amostras por janela e um Random Forest com acurácia de teste de 98% e média de validação cruzada de cinco folds de aproximadamente 97,73%. Esses números descrevem somente a avaliação executada no conjunto sintético atual; precisam ser reavaliados após revisão da simulação, da geração dos dados e do protocolo de separação treino/teste.

## Limitações importantes

- O gerador de exemplos sintéticos e a simulação física precisam ser verificados quanto à coerência e à ligação entre si.
- As definições de `Surto Leve` e `Surto Severo` são operacionais e devem ser justificadas; não devem ser tratadas como classes universais sem referência adequada.
- A topologia elétrica simplificada, os parâmetros e a interpretação da tensão de entrada/saída precisam ser documentados e validados.
- O resultado de classificação não demonstra generalização para sinais reais.
- A implementação embarcada e os requisitos de aquisição/condicionamento do sinal ainda precisam ser especificados e testados.

## Segurança e escopo

Este repositório documenta uma pesquisa acadêmica experimental. Não use o protótipo para comandar proteção real, conectar diretamente o ESP32 a uma rede de média tensão ou tomar decisões operacionais em instalações elétricas. Medições em redes elétricas exigem instrumentação isolada, dimensionamento e supervisão profissional apropriados.

## Organização do trabalho

Consulte `CONTEXTO_PROJETO.md`, `DECISOES.md` e `PROXIMOS_PASSOS.md` antes de alterar a metodologia. Registre mudanças relevantes e os resultados de cada experimento.
