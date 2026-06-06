# Detecção de Placas Veiculares com YOLOv11 e HOG + SVM

Projeto desenvolvido para a disciplina de **Machine Learning**, com o objetivo de comparar uma abordagem baseada em **Deep Learning** (YOLOv11) e uma abordagem clássica de **Machine Learning** (HOG + SVM) para detecção e reconhecimento de placas veiculares.

---

## Objetivo

Desenvolver um sistema capaz de:

- Detectar automaticamente a região da placa em uma imagem de veículo;
- Realizar o recorte da Região de Interesse (ROI);
- Comparar o desempenho entre dois modelos distintos:
  - **YOLOv11** (Detecção de Objetos)
  - **HOG + SVM** (Classificação de Imagens)

Além disso, o projeto conta com uma interface desenvolvida em **Streamlit**, permitindo que o usuário envie uma imagem e visualize os resultados produzidos pelos dois modelos.

---

## Dataset

Foi utilizado um dataset contendo imagens reais de veículos com suas respectivas anotações no formato YOLO.

### Estrutura do conjunto de dados

| Conjunto | Quantidade |
|----------|------------|
| Treinamento | 4.292 imagens |
| Validação | 391 imagens |
| Teste | 276 imagens |

Para o treinamento do HOG + SVM, foram gerados automaticamente recortes positivos (placa) e negativos (não placa) a partir das anotações do YOLO.

---

## Modelos Utilizados

### YOLOv11

Modelo baseado em Deep Learning utilizado para localizar automaticamente a placa na imagem completa.

**Função principal:**
- Detectar a posição da placa;
- Gerar a Bounding Box;
- Realizar o recorte da região detectada.

### HOG + SVM

Abordagem clássica de Machine Learning composta por:

- Extração de características utilizando HOG (Histogram of Oriented Gradients);
- Classificação utilizando Support Vector Machine (SVM).

Neste projeto, o HOG + SVM recebe como entrada o recorte gerado pelo YOLO e classifica a região como:

- Placa
- Não placa

---

## Fluxo do Sistema

```
Imagem de entrada
        │
        ▼
    YOLOv11
(Localiza a placa)
        │
        ▼
Recorte da Região
        │
        ▼
   HOG + SVM
(Classifica o recorte)
        │
        ▼
Exibição dos resultados
```

---

## Resultados Obtidos

### YOLOv11

| Métrica | Resultado |
|----------|-----------|
| Precision | 99,85% |
| Recall | 99,28% |
| mAP@50 | 99,50% |
| mAP@50-95 | 82,63% |

---

### HOG + SVM

| Métrica | Resultado |
|----------|-----------|
| Acurácia | 99,82% |
| Precisão | 100,00% |
| Recall | 99,64% |
| F1-Score | 99,82% |

### Matriz de Confusão do HOG + SVM

|               | Predito Não Placa | Predito Placa |
|---------------|------------------|---------------|
| Real Não Placa | 270 | 0 |
| Real Placa | 1 | 275 |

Total de amostras avaliadas: **546**

Apenas **uma classificação incorreta** foi observada no conjunto de teste.

---

## Estrutura do Projeto

```
plate-detection-ml/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── best.pt
│   └── svm_model.pkl
│
├── notebooks/
│   └── treinamento_kaggle.ipynb
│
├── results/
│   ├── comparacao_modelos.csv
│   ├── resultados_yolo.csv
│   ├── metricas_svm.csv
│   ├── confusion_matrix_svm.png
│   ├── results.csv
│   ├── results.png
│   ├── confusion_matrix.png
│   ├── BoxF1_curve.png
│   ├── BoxP_curve.png
│   ├── BoxPR_curve.png
│   └── BoxR_curve.png
│
└── examples/
```

---

## Tecnologias Utilizadas

- Python 3
- YOLOv11 (Ultralytics)
- OpenCV
- Scikit-Learn
- Scikit-Image
- NumPy
- Pandas
- Matplotlib
- Joblib
- Streamlit

---

## Executando o Projeto

### 1. Clone o repositório

```bash
git clone https://github.com/SEU-USUARIO/plate-detection-ml.git
cd plate-detection-ml
```

### 2. Instale as dependências

```bash
pip install -r requirements.txt
```

### 3. Execute a aplicação

```bash
streamlit run app.py
```

---

## Aplicação Web

A interface desenvolvida em Streamlit permite:

- Upload de uma imagem;
- Detecção automática da placa utilizando YOLOv11;
- Exibição do recorte gerado;
- Classificação do recorte pelo HOG + SVM;
- Comparação entre as duas abordagens.

---

## Autores

Projeto desenvolvido para a disciplina de **Machine Learning**.

**Aluno:**
João Marcos dos Santos Gil

---

## Licença

Este projeto possui finalidade exclusivamente acadêmica.
