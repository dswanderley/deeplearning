# DCGAN: Deep Convolutional Generative Adversarial Networks

Este projeto implementa uma DCGAN (Deep Convolutional GAN) utilizando PyTorch para gerar imagens a partir do dataset CIFAR-10. A DCGAN é uma extensão das GANs tradicionais, utilizando redes convolucionais profundas, tanto para o gerador quanto para o discriminador, o que a torna especialmente eficaz na geração de imagens de alta qualidade.

## Estrutura do Projeto

- **models.py**: Contém a definição das classes `Generator` e `Discriminator`, que implementam o gerador e o discriminador da DCGAN utilizando camadas convolucionais (`Conv2d` e `ConvTranspose2d`).
- **train.py**: Script que encapsula o treinamento da DCGAN. O script carrega o dataset CIFAR-10, realiza o treinamento da rede, e salva o modelo do gerador com a menor perda (best model).
- **inference.py**: Script utilizado para gerar e visualizar novas imagens utilizando o gerador treinado. Carrega o modelo salvo e realiza a inferência para gerar imagens a partir de vetores de ruído aleatórios.

## Estrutura da DCGAN

1. **Gerador (Generator)**:
   - O gerador usa camadas convolucionais transpostas para expandir um vetor de ruído de baixa dimensionalidade (normalmente de tamanho 100) em uma imagem de alta dimensionalidade (64x64 pixels, 3 canais para imagens RGB).
   - A saída do gerador é uma imagem normalizada no intervalo [-1, 1], utilizando a função de ativação `Tanh`.

2. **Discriminador (Discriminator)**:
   - O discriminador utiliza camadas convolucionais tradicionais para processar imagens de entrada e classificar se elas são reais (provenientes do dataset) ou falsas (geradas pelo gerador).
   - A função de ativação final é uma sigmoide (`Sigmoid`), que produz uma probabilidade (0 para falso e 1 para real).

## Fluxo do Treinamento

1. **Inicialização**:
   - As redes gerador e discriminador são inicializadas, e a função de perda utilizada é a Binary Cross-Entropy (`BCELoss`).
   - O gerador é otimizado para enganar o discriminador, gerando imagens que se aproximam das imagens reais do dataset.
   - O discriminador é otimizado para distinguir corretamente entre imagens reais e imagens geradas.

2. **Treinamento**:
   - A cada época, o discriminador é treinado primeiro com imagens reais e, em seguida, com imagens falsas geradas.
   - O gerador é então treinado com o objetivo de enganar o discriminador, tentando produzir imagens que sejam classificadas como reais.
   - O modelo do gerador com a menor perda ao longo do treinamento é salvo como o melhor modelo.

3. **Inferência**:
   - Após o treinamento, o modelo gerador pode ser carregado para gerar novas imagens, utilizando vetores de ruído aleatórios como entrada.

## Dataset Utilizado

O dataset utilizado no projeto é o **CIFAR-10**, um conjunto de dados de imagens coloridas contendo 60.000 imagens em 10 classes diferentes. As imagens são redimensionadas para 64x64 pixels para serem compatíveis com a arquitetura da DCGAN.

## Como Funciona

1. O **Gerador** recebe como entrada um vetor de ruído aleatório e gera uma imagem sintética a partir dele.
2. O **Discriminador** recebe uma imagem (real ou gerada) e tenta classificá-la como real ou falsa.
3. Durante o treinamento, o objetivo do gerador é enganar o discriminador para que classifique as imagens geradas como reais, enquanto o discriminador tenta melhorar sua capacidade de detectar imagens falsas.

## Objetivo

O objetivo principal da DCGAN é treinar um gerador capaz de produzir imagens realistas que são indistinguíveis das imagens reais para o discriminador. Isso é conseguido através de um processo de treinamento adversarial entre o gerador e o discriminador, onde ambos melhoram continuamente suas habilidades.

## Resultados

O modelo treinado pode gerar novas imagens com base em vetores de ruído aleatórios. Essas imagens imitam as características do dataset CIFAR-10, produzindo exemplos variados de objetos como aviões, carros, pássaros e outros.

## Aplicações

As DCGANs têm várias aplicações em áreas como:
- **Geração de Imagens Realistas**: Podem ser usadas para criar novas imagens com base em um dataset de treinamento.
- **Estilo de Transferência**: Utilizando DCGANs, é possível gerar imagens em diferentes estilos artísticos.
- **Super-Resolução de Imagens**: Podem ser aplicadas para melhorar a resolução de imagens de baixa qualidade.
