# Deep Learning Project

Este repositório contém diversos tópicos de **Deep Learning**, com foco inicial em **Redes Adversariais Generativas (GANs)**, incluindo **GANs tradicionais** e **DCGANs (Deep Convolutional GANs)**. O projeto está estruturado de forma modular e escalável, permitindo fácil expansão para outros tópicos de aprendizado profundo.

## Visão Geral

A ideia deste projeto é servir como um ambiente de exploração e implementação de diferentes modelos de deep learning. Começamos com GANs, mas o objetivo é expandir para uma ampla gama de modelos e técnicas, como **Autoencoders Variacionais (VAE)**, **Redes Neurais Convolucionais (CNN)**, **Redes Recorrentes (RNN)**, e muito mais.

### Principais Componentes:

- **Módulos por Tópico**: Cada técnica ou modelo é isolado em seu próprio módulo, o que garante que o código seja fácil de manter e expandir. Atualmente, temos implementações de **GANs** e **DCGANs**.
- **Arquivos de Configuração Centralizados**: As configurações de treinamento e inferência são definidas em arquivos dedicados (YAML ou JSON), permitindo ajustes fáceis sem a necessidade de modificar o código-fonte.
- **Dados Estruturados**: O repositório já inclui referências para os datasets **CIFAR-10** e **MNIST**, que são utilizados para treinamento e testes dos modelos. A estrutura está pronta para adicionar novos conjuntos de dados conforme necessário.

### Estrutura de Pastas:

- **data/**: Contém os datasets utilizados para treinamento e inferência. Atualmente, temos os datasets **CIFAR-10** e **MNIST** organizados em subdiretórios.
- **models/**: Diretório que armazena os pesos dos modelos treinados. Aqui são salvos os checkpoints e as versões finais dos geradores para GANs e DCGANs.
- **src/**: O diretório principal de código-fonte, onde cada modelo é organizado em seu próprio módulo. Cada subdiretório (como `gan` ou `dcgan`) contém scripts de treinamento, inferência, definição de modelos e documentação específica.
- **logs/**: Diretório onde serão armazenados os logs de treinamento, que incluem métricas como perda e precisão, e podem ser utilizados com ferramentas como **TensorBoard**.

### Modelos Implementados:

- **GAN (Generative Adversarial Network)**:
  - Implementação básica de GANs, onde um **Gerador** tenta criar imagens falsas e um **Discriminador** tenta distinguir essas imagens das reais. Este módulo usa o dataset MNIST para demonstrar a criação de imagens de dígitos escritos à mão.

- **DCGAN (Deep Convolutional GAN)**:
  - Um modelo GAN que utiliza camadas convolucionais profundas, particularmente eficiente para a geração de imagens realistas. O DCGAN usa o dataset CIFAR-10, com imagens coloridas de 64x64 pixels, e é otimizado para gerar imagens com características visuais mais detalhadas.

### Expansão e Futuro

Embora o projeto comece com GANs e DCGANs, o objetivo é adicionar novos tópicos de Deep Learning, como:

- **Autoencoders Variacionais (VAE)**: Modelos generativos que podem capturar a distribuição latente dos dados.
- **Redes Neurais Convolucionais (CNN)**: Para tarefas de classificação e detecção de objetos.
- **Redes Recorrentes (RNN)**: Para tarefas de séries temporais e processamento de linguagem natural (NLP).
- **Transfer Learning**: Aproveitar redes pré-treinadas para melhorar o desempenho em datasets específicos.
- **Redes Neurais de Atenção**: Como Transformers, que revolucionaram o campo do NLP.

Este projeto será continuamente expandido para explorar novas áreas de pesquisa e inovação em aprendizado profundo.

### Como Contribuir

Sinta-se à vontade para contribuir com novos modelos, otimizações ou qualquer sugestão que ajude a expandir o projeto. A modularidade da estrutura facilita a adição de novos tópicos e técnicas.

---

**Nota**: Este README está em constante evolução, à medida que novos módulos e tópicos são adicionados ao repositório.
