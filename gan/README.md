# Generative Adversarial Network (GAN) for MNIST

Este projeto implementa uma GAN simples utilizando PyTorch para gerar imagens de dígitos escritos à mão, usando o dataset MNIST. A implementação está dividida em três blocos: model, train e inference.

## Requisitos

Para rodar este projeto, é necessário ter instalado:
- Python 3.7 ou superior
- PyTorch
- Torchvision
- Matplotlib

## Arquitetura

### 1. Model

Neste bloco, definimos as duas redes principais da GAN:

- **Gerador (Generator)**: O gerador transforma um vetor de ruído (normalmente amostras de uma distribuição normal) em uma imagem de 28x28, que simula um dígito escrito à mão.
- **Discriminador (Discriminator)**: O discriminador é responsável por distinguir entre imagens reais (do dataset MNIST) e imagens geradas pelo gerador.

Ambos os modelos são construídos usando camadas totalmente conectadas (fully connected) e funções de ativação adequadas. O gerador utiliza a função de ativação `Tanh()` para garantir que a saída esteja no intervalo de -1 a 1, enquanto o discriminador usa `Sigmoid()` para produzir uma probabilidade de a imagem ser real ou falsa.

### 2. Train

Este bloco contém o loop de treinamento da GAN:

- **Passo do Discriminador**: O discriminador é treinado tanto em imagens reais quanto em imagens falsas geradas pelo gerador. O objetivo é maximizar sua capacidade de distinguir entre as duas.
  - Para as imagens reais, a perda é calculada utilizando a função `BCELoss` com rótulos reais (1).
  - Para as imagens falsas, a perda é calculada com rótulos falsos (0).
  - O discriminador é então atualizado para melhorar sua precisão.

- **Passo do Gerador**: O gerador é treinado para enganar o discriminador. O objetivo do gerador é maximizar a probabilidade de o discriminador classificar suas imagens como reais.
  - O gerador é atualizado para minimizar a perda calculada contra o discriminador.

O processo de treinamento continua por várias épocas, ajustando continuamente ambos os modelos.

### 3. Inference (ou Evaluation)

Após o treinamento, o gerador pode ser utilizado para criar novas imagens. Este bloco permite gerar amostras de imagens utilizando o gerador treinado. As imagens são criadas a partir de vetores de ruído aleatórios e visualizadas utilizando a biblioteca Matplotlib.

## Estrutura do Código

- **Model**: Define a arquitetura das redes (Gerador e Discriminador).
- **Train**: Executa o treinamento da GAN, alternando entre o treinamento do discriminador e do gerador.
- **Inference (Evaluation)**: Gera e visualiza novas imagens após o treinamento.

## Funcionamento

1. **Gerador**: Gera imagens sintéticas a partir de ruído aleatório.
2. **Discriminador**: Avalia as imagens e classifica como real ou falsa.
3. **Treinamento**: Ambos os modelos são treinados em conjunto, com o gerador tentando melhorar a qualidade das imagens e o discriminador tentando melhorar sua capacidade de distinção.

## Personalização

Você pode ajustar alguns parâmetros para experimentar com o treinamento:
- **latent_size**: O tamanho do vetor de ruído de entrada do gerador.
- **hidden_size**: O tamanho das camadas ocultas tanto no gerador quanto no discriminador.
- **num_epochs**: O número de épocas de treinamento.
- **learning_rate**: A taxa de aprendizado utilizada pelos otimizadores.


### Salvamento e Carregamento do Modelo

Após o treinamento, o modelo gerador é salvo em um arquivo `.pth` utilizando o PyTorch. Para carregar o modelo e utilizá-lo para gerar novas imagens na fase de inferência, os seguintes passos são executados:

- **Salvamento do Modelo**: O modelo do gerador é salvo após o término do treinamento.
- **Carregamento do Modelo**: Durante a inferência, o modelo salvo é carregado e usado para gerar novas imagens a partir de ruído aleatório.

O arquivo `generator.pth` armazena os pesos do modelo treinado e é carregado utilizando `torch.load` para avaliar o gerador.


## Observações

Este projeto foi desenvolvido para ser simples e direto, ideal para quem está começando a trabalhar com GANs. Se desejar estender o projeto, considere experimentar arquiteturas diferentes (como convolucionais) ou datasets mais complexos.

