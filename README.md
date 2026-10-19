# Projeto modular — TCC de comparação de modelos YOLO

Projeto para executar validação cruzada K-Fold com YOLOv8s, YOLOv10s e YOLO26s, reutilizando exatamente os mesmos folds para todos os modelos.

## 1. Requisitos

- Python 3.10 ou superior
- Dataset YOLO com imagens e rótulos
- Um arquivo `data.yaml` configurado corretamente
- PyTorch/Ultralytics compatíveis com sua GPU e com os modelos utilizados

Instale as dependências:

```bash
pip install -r requirements.txt
```

## 2. Configure o dataset

Abra `config/settings.py` e ajuste `DATA_YAML` para o caminho do seu `data.yaml`.

Exemplo de `data.yaml`:

```yaml
path: /caminho/para/D-Fire
train: images/train
val: images/val
test: images/test
names:
  0: smoke
  1: fire
```

O script combina as imagens dos conjuntos `train` e `val` para criar os 5 folds. O conjunto `test` fica separado e não participa do treinamento nem da escolha do melhor modelo.

As imagens e rótulos devem seguir a estrutura YOLO padrão, por exemplo:

```text
D-Fire/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
└── data.yaml
```

Os diretórios podem ter outros nomes, desde que estejam indicados corretamente no YAML. Se `train` e `val` apontarem para arquivos `.txt`, cada linha deve conter o caminho de uma imagem.

## 3. Executar

Na raiz do projeto:

```bash
python main.py
```

Por padrão, o projeto usa 5 folds, seed 42, 100 épocas, imagem 640, batch 24, patience 20 e device `0`. Ajuste esses valores em `config/settings.py`.

## 4. Saídas

Os arquivos serão gravados em `runs/kfold_tcc/`:
- `folds/`: arquivos com as imagens de treino e validação de cada fold e YAMLs correspondentes;
- `results_per_fold.csv`: métricas de cada modelo em cada fold;
- `summary.csv`: média e desvio-padrão das métricas por modelo;
- `test_results.csv`: avaliação final no conjunto de teste, executada uma vez por modelo usando o melhor checkpoint escolhido no processo de validação cruzada.

**Importante:** o teste final é uma avaliação externa ao K-Fold. O script treina cada modelo em cada fold e seleciona, para cada arquitetura, o checkpoint com maior mAP50-95 de validação entre os folds; depois avalia esse checkpoint no teste. Para um experimento metodologicamente mais forte, pode ser preferível treinar novamente cada arquitetura com todos os dados de desenvolvimento (train+val) usando uma configuração definida a priori, e só então avaliar uma única vez no teste. A estratégia adotada deve ser explicada no TCC.
