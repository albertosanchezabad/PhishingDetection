# Empirical Analysis and Comparative Study of Automated Phishing Detection Techniques in Email Communications

This repository contains the complete implementation and evaluation of multiple techniques for automated phishing detection in email communications. The project compares three main approaches: heuristic-based rules, traditional machine learning algorithms (SVM, Random Forest, Naive Bayes, Logistic Regression), and deep learning models (BERT, LSTM, BiLSTM).

## 🎯 Key Features

- **Comprehensive Dataset**: 218,086 email records across 11 datasets spanning 24 years (1998-2022)
- **Autonomous System**: Completely self-contained detection without external service dependencies  
- **High Performance**: Achieved 99.31% F1-score with BiLSTM model
- **Reproducible Research**: Full implementation with hyperparameter optimization and detailed documentation

## 📊 Results

Deep learning models achieved superior performance (>98% F1-score), followed by traditional ML algorithms (>95% F1-score), while heuristic approaches showed significant limitations in generalization across diverse phishing types.

## 📂 Dataset Access

The datasets used in this research are not included in this repository due to size constraints. You can download the complete "Phishing Email Curated Datasets" collection from the original source at Zenodo: 

🔗 https://zenodo.org/records/10091756

## 🚀 Execution Order

The logical execution order of scripts is as follows, assuming you have the initial datasets in a `/dataset/iniciales` folder:

### 1. Data Preprocessing

1. **`preprocesamiento_general.py`** - General cleaning for all datasets

2. **`limpiar_idiomas.py`** - Keep only English emails

3. **`diagnostico.py`** - Review cleaning evolution and language filtering

4. **`visualizar_datos.py`** - View balancing and examples in datasets from each folder (modify code to target specific dataset folders: iniciales, preprocesamiento_general, or solo_ingles)

5. **`conseguir_grupos.py`** - Explore how to obtain combined datasets

6. **`agrupar_datasets_solo_ingles.py`** - Group into combined datasets

### 2. Model Implementation and Evaluation

7. **`heuristicas.py`** - Heuristic implementation and evaluation

8. **`preprocesamiento_ML.py`** - ML preprocessing (without combined groups to avoid repeated preprocessing of same records)

9. **`agrupar_datasets_preprocesados_ML.py`** - Group into combined datasets with all records already preprocessed

10. **`tf-idf_ML.py`** - Apply TF-IDF for ML

11. **`ML.py`** - Train, optimize and evaluate ML models

### 3. Deep Learning Models

12. **`preparar_BERT.py`** - Prepare data for BERT

13. **`BERT_HT.py`** - Train, optimize and evaluate BERT

14. **`preparar_LSTM.py`** - Prepare data for LSTM (and BiLSTM)

15. **`LSTM_KT.py`** - Train, optimize and evaluate LSTM

16. **`BiLSTM_KT.py`** - Train, optimize and evaluate BiLSTM

## 💻 Usage

1. Download the datasets from the Zenodo link provided above

2. Place the initial datasets in `/dataset/iniciales` folder

3. Execute the scripts in the order specified above

4. Each script generates intermediate results that are used by subsequent scripts

**Note**: This research provides practical insights for cybersecurity professionals and serves as a foundation for developing robust, scalable phishing detection systems in enterprise environments.
