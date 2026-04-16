# DS_Proto-TE

An interpretable clinical prediction framework that combines **Prototype Networks** with **Temporal Encoding** for Electronic Health Record (EHR) data. Built on the MIMIC dataset, this project implements a progressive training strategy to learn clinically meaningful prototypes for ICU outcome prediction.

## Overview

DS_Proto-TE addresses the challenge of building **interpretable** deep learning models for clinical decision support. Instead of relying on black-box predictions, the model learns a set of **prototypes** (representative patient patterns) and makes predictions based on similarity to these prototypes, enabling clinicians to understand *why* a prediction was made.

### Key Features

- **Progressive (Evolutionary) Training**: Multi-stage training pipeline where each stage builds upon pretrained weights from the previous stage
- **Channel-wise / Category-wise Temporal Encoding**: Independent LSTM/GRU/Transformer encoders for each clinical variable or variable category, capturing variable-specific temporal dynamics
- **Prototype-based Prediction**: Syndrome prototypes (SynProtoLayer) and demographic prototypes (DemoProtoLayer) for interpretable classification
- **Multi-Head Attention**: Custom QKV attention mechanism for learning inter-variable relationships
- **Multiple Encoder Backends**: Supports LSTM, GRU, Transformer, and hybrid architectures

## Architecture

```
Input EHR Data (N, 48, 76)
         |
   Temporal Encoder (Category-wise LSTM/GRU/Transformer)
         |
   Multi-Head Attention (Variable Attention)
         |
   Prototype Layer (Syndrome / Demographic)
         |
   Classification Head --> Binary Prediction
```

### Training Stages

| Stage | Description | Trainable Components |
|-------|-------------|----------------------|
| **Base0** | Train temporal encoder + classifier from scratch | All |
| **Base1** | Freeze Base0 encoder, train demographic encoder | DemoEncoder, cls_base1 |
| **Proto0** | Freeze encoder, train syndrome prototype layer | SynProtoLayer, cls_syn_proto |
| **Proto2** | Similar to Proto0, with consistency loss | SynProtoLayer, cls_syn_proto |
| **Proto1** | Train demographic prototype layer (freeze Syn) | DemoProtoLayer, cls_demo_proto |
| **Final** | Joint fine-tuning of all components | All (or selected) |

## Project Structure

```
DS_Proto-TE/
|-- Configs.py              # Path configurations & clinical channel mappings
|-- Encoders.py             # Temporal encoders, attention modules
|-- EvoPNet.py              # Full prediction model (LSTM-based)
|-- EvoPNet_TF.py           # Full prediction model (Transformer-based)
|-- Main_MIMIC.py           # Main training script with CLI arguments
|-- main.py                 # Standalone encoder pretraining script
|-- Utils_basic.py          # Evaluation metrics (ACC, AUROC, AUPRC)
|-- Utils_embed.py          # Dataset & DataLoader utilities
|-- Utils_eval.py           # Model evaluation functions
|-- Utils_train.py          # Model initialization & training loop
|-- MIMIC_dataset_demo/     # Demo dataset (demographic features only)
|-- encoder_only.pth        # Pretrained encoder weights
|-- LICENSE                 # MIT License
```

## Data

The project uses the **MIMIC** (Medical Information Mart for Intensive Care) dataset:

- **EMR time-series data**: Shape `(N, 48, 76)` -- 48 time steps, 76 features
- **Demographic data**: Shape `(N, D_demo)` -- ethnicity, sex, age, etc.
- **Labels**: Binary (e.g., in-hospital mortality)

### Clinical Variable Categories (17 groups, 76 features)

The 76 input features are organized into 17 clinically meaningful categories:

| Category | Features |
|----------|----------|
| Capillary refill rate | 3 features |
| Diastolic blood pressure | 2 features |
| Fraction inspired oxygen | 2 features |
| Glasgow Coma Scale (eye/motor/total/verbal) | 4 groups, 51 features total |
| Glucose | 2 features |
| Heart rate | 2 features |
| Height | 2 features |
| Mean blood pressure | 2 features |
| Oxygen saturation | 2 features |
| Respiratory rate | 2 features |
| Systolic blood pressure | 2 features |
| Temperature | 2 features |
| Weight | 2 features |
| pH | 2 features |

## Requirements

- Python 3.8+
- PyTorch 1.10+
- NumPy
- scikit-learn
- pandas
- matplotlib
- seaborn
- tqdm

## Usage

### 1. Data Preparation

Place the MIMIC dataset files in your data directory:

```
<data_dir>/
|-- Train_data.npy      # Training EMR data
|-- Train_demo.npy      # Training demographic data
|-- Train_label.npy     # Training labels
|-- Val_data.npy        # Validation EMR data
|-- Val_demo.npy        # Validation demographic data
|-- Val_label.npy       # Validation labels
|-- Test_data.npy       # Test EMR data
|-- Test_demo.npy       # Test demographic data
|-- Test_label.npy      # Test labels
```

Update the data path in `Main_MIMIC.py`:

```python
MIMIC_data_dir = '<your_data_dir>'
```

### 2. Training

Train the model in progressive stages:

```bash
# Stage 1: Base temporal encoder
python Main_MIMIC.py Base0 --TempEnc_type Cat-LSTM --EMR_emb 8 --epochs 100

# Stage 2: Demographic encoder
python Main_MIMIC.py Base1 --TempEnc_type Cat-LSTM --EMR_emb 8 --DEMO_emb 64 --epochs 100
```

### Key Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `Model_type` | (required) | Training stage: Base0, Base1, Proto0, Proto1, Proto2, Final |
| `--TempEnc_type` | Cat-LSTM | Encoder type: LSTM, CW-LSTM, Cat-LSTM, TF, Cat-Transformer |
| `--EMR_emb` | 8 | EMR embedding dimension |
| `--DEMO_emb` | 64 | Demographic embedding dimension |
| `--syn_k` | 50 | Number of syndrome prototypes |
| `--demo_k` | 20 | Number of demographic prototypes |
| `--bs` | 32 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--p` | 0.3 | Dropout rate |
| `--epochs` | 100 | Maximum training epochs |
| `--early_stop` | 10 | Early stopping patience |
| `--nlayers` | 2 | Number of encoder layers |
| `--nhead` | 4 | Number of attention heads |

### 3. Standalone Encoder Pretraining

To pretrain the channel-wise encoder with a reconstruction objective:

```bash
python main.py
```

## Evaluation Metrics

The model is evaluated using:

- **Accuracy (ACC)**
- **Area Under the ROC Curve (AUROC)**
- **Area Under the Precision-Recall Curve (AUPRC)**

Results are saved as CSV files and loss curves are plotted as PNG images.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
