# Relatório de Análise de Experimentos de Aprendizado por Reforço Profundo

## Introdução

Este relatório apresenta uma análise dos experimentos realizados com os métodos de Aprendizado por Reforço Profundo (Deep Q-learning e Double Deep Q-learning) em diferentes contextos. Foram utilizados ambientes do Classic Control do Gymnasium e ambientes do Atari para avaliar o desempenho dos algoritmos. A análise experimental considera o projeto das redes neurais, hiperparâmetros, métodos de otimização e critérios de parada utilizados em cada experimento.

## Ambientes e Configurações

### Classic Control do Gymnasium
Diversos variações houveram porém as melhores seja para o Cart pole ou seja para o mountain car se mantiveram nas mesmas configurações.
#### Ambiente 1: Mountain Car
- Rede Neural: Camadas lineares com o PyTorch.
- Topologia da Rede: 
    - Camada com entrada do tamanho do input e saída de 128 neurônios e ativação relu.
    - Camada com entrada de 128 e saída de 128 neurônios e ativação relu.
    - Camada com entrada de 128 e saída com tamanho do output e ativação relu.
- Método de Otimização: Adam
- Hiperparâmetros: 
    - Episódios : 10000
    - Episilion inicial/final: 1/0.05
    - Learning Rate: 0.0001
    - Discount Factor: 0.99
    - Double Q ? Sim
- Critérios de Parada: Nenhum

**10% do treinamento**

<video controls width="500">
    <source src="classic/results/MountainCar-v0-10000-100-5-100-99-True/10.mp4" type="video/mp4">
    Desculpe, mas não foi possível carregar o vídeo.
</video>


**50% do treinamento**

<video controls  width="500">
    <source src="classic/results/MountainCar-v0-10000-100-5-100-99-True/50.mp4" type="video/mp4">
    Desculpe, mas não foi possível carregar o vídeo.
</video>


**Final do Treinamento**

<video controls  width="500">
    <source src="classic/results/MountainCar-v0-10000-100-5-100-99-True/100.mp4" type="video/mp4">
    Desculpe, mas não foi possível carregar o vídeo.
</video>

** Gráficos do Resultado **
<img src="classic/bestmountain.png" alt="bestmountain">       
 
#### Ambiente 2: Mountain Car
- Rede Neural: Camadas lineares com o PyTorch.
- Topologia da Rede: [INSERIR DETALHES DA TOPOLOGIA DA REDE AQUI, COMO NÚMERO DE CAMADAS E NEURÔNIOS EM CADA CAMADA].
- Função de Ativação: [INSERIR FUNÇÃO DE ATIVAÇÃO UTILIZADA, EX: ReLU, Sigmoid].
- Método de Otimização: [INSERIR MÉTODO DE OTIMIZAÇÃO UTILIZADO, EX: Adam, RMSprop].
- Hiperparâmetros: [INSERIR VALORES DOS HIPERPARÂMETROS UTILIZADOS, EX: Taxa de Aprendizado, Tamanho do Lote].
- Critérios de Parada: [INSERIR CRITÉRIOS DE PARADA UTILIZADOS, EX: Número máximo de episódios, Convergência do Desempenho].

### Ambientes do Atari

#### Ambiente 3: Space Invaders
- Rede Neural: Camadas convolucionais e lineares com o PyTorch.
- Topologia da Rede: [INSERIR DETALHES DA TOPOLOGIA DA REDE AQUI, COMO NÚMERO DE CAMADAS E NEURÔNIOS EM CADA CAMADA].
- Função de Ativação: [INSERIR FUNÇÃO DE ATIVAÇÃO UTILIZADA, EX: ReLU, Sigmoid].
- Método de Otimização: [INSERIR MÉTODO DE OTIMIZAÇÃO UTILIZADO, EX: Adam, RMSprop].
- Hiperparâmetros: [INSERIR VALORES DOS HIPERPARÂMETROS UTILIZADOS, EX: Taxa de Aprendizado, Tamanho do Lote].
- Critérios de Parada: [INSERIR CRITÉRIOS DE PARADA UTILIZADOS, EX: Número máximo de episódios, Convergência do Desempenho].

#### Ambiente 4: [INSERIR NOME DO AMBIENTE ESCOLHIDO]
- Rede Neural: Camadas convolucionais e lineares com o PyTorch.
- Topologia da Rede: [INSERIR DETALHES DA TOPOLOGIA DA REDE AQUI, COMO NÚMERO DE CAMADAS E NEURÔNIOS EM CADA CAMADA].
- Função de Ativação: [INSERIR FUNÇÃO DE ATIVAÇÃO UTILIZADA, EX: ReLU, Sigmoid].
- Método de Otimização: [INSERIR MÉTODO DE OTIMIZAÇÃO UTILIZADO, EX: Adam, RMSprop].
- Hiperparâmetros: [INSERIR VALORES DOS HIPERPARÂMETROS UTILIZADOS, EX: Taxa de Aprendizado, Tamanho do Lote].
- Critérios de Parada: [INSERIR CRITÉRIOS DE PARADA UTILIZADOS, EX: Número máximo de episódios, Convergência do Desempenho].

## Resultados e Análises

[INSERIR AQUI UMA ANÁLISE DETALHADA DOS RESULTADOS OBTIDOS EM CADA AMBIENTE, COMPARANDO OS MÉTODOS DE APRENDIZADO POR REFORÇO PROFUNDO UTILIZADOS, EXPLICANDO O GRAU DE IMPORTÂNCIA DAS CONCLUSÕES OBTIDAS].

## Gráficos

[INSERIR AQUI OS GRÁFICOS QUE REPRESENTAM O DESEMPENHO DOS MÉTODOS DE APRENDIZADO POR REFORÇO PROFUNDO EM CADA AMBIENTE. EX: GRÁFICOS DE RECOMPENSAS POR EPISÓDIO, GRÁFICOS DE TAXA DE APRENDIZADO AO LONGO DO TREINAMENTO, ETC.].

## Códigos Implementados

Para acessar os códigos implementados para os experimentos, utilize os links abaixo:

- [Link para o código do ambiente Cart Pole](https://exemplo.com/codigo_cart_pole)
- [Link para o código do ambiente Mountain Car](https://exemplo.com/codigo_mountain_car)
- [Link para o código do ambiente Space Invaders](https://exemplo.com/codigo_space_invaders)
- [Link para o código do ambiente escolhido](https://exemplo.com/codigo_ambiente_escolhido)

## Conclusão

[INSERIR AQUI UMA CONCLUSÃO GERAL DOS RESULTADOS OBTIDOS, DESTACANDO AS DIFERENÇAS ENTRE OS MÉTODOS DE APRENDIZADO POR REFORÇO PROFUNDO E SUAS IMPLICAÇÕES NOS DIFERENTES AMBIENTES TESTADOS].

Espero que este template ajude você a estruturar o seu relatório em markdown. Lembre-se de preencher as informações específicas de cada experimento e ambiente para tornar o relatório completo e informativo. Boa sorte!