# Sentiment Analysis API

## Overview

This project provides a simple & flexible implementation of a sentiment classifier utilizing Twitter dataset (~1.6M records). Additionally, the application exposes a RESTful API endpoint for real-time _synchronous_ sentiment evaluation.

## Project Tree 

```
sentiment-analysis/
├── config.py                         # Global configuration and some useful helpers
├── main.py                           # FastAPI application 
├── prepare.py                        # Contains preliminary files for data and vocabulary building
│
├── data/                            
│   ├── tokenizers.py                 # Tokenizer implementations (Tweet, BPE, GPT-2)
│   ├── utils.py                      # Data loading and preprocessing utilities
│   ├── eda.ipynb                     
│   ├── toks/                         # Trained tokenizer files
│   │   ├── bpe.json                  
│   │   ├── bpe_vocab.json            
│   │   ├── gpt2_vocab.json           
│   │   └── tweet_vocab.json                        
│
├── training/                         
│   ├── train.py                      
│   ├── evaluate.py                   
│   ├── models/                       
│   │   └── transformer.py            
│
├── tests/                            # Application testing utility (not the model)
│   ├── __init__.py
│   ├── conftest.py                   
│   └── test_performance.py           # Performance and integration tests
```


## Installation
After cloning the repository and navigating to the _root_ of the project, execute the following commands 
```
chmod +x setup.sh 
sh setup.sh  # ./setup.sh 
```
<br>

## Training (Optional)

This project has a provision for a custom training pipeline with 3 different tokenizers

#### Tokenizers
- **TweetTokenizer**: 
   - Optimized for [Twitter](https://medium.com/@aminbaybon/tokenization-using-nltk-tweettokenizer-d1213c1412d9) text preprocessing
   - Trivial choice for this task as it provides direct way to eliminate the most frequently used non-alphanumeric characters like `@, #,` and tokenizations for emoticons (`:), 8))
   - If run, builds a vocabulary of size `30k` which covers most of the twitter dataset. OOV words are handled
   using the `[UNK]` token
   - Standalone `Tweettokenizer()` is data dependent and it grows in size as more words are added to the vocab. `SnowballStemmer` support is provided for better grouping of tokens
   
- **BPETokenizer**: 
   - Handles misspellings and typos better than word-level tokenization `(Tweettokenizer)`
   - OOV (out-of-vocabulary) rate is reduced
   - More efficient vocabulary usage; represents more words with fewer tokens (vocab size of `30k` should have a really good coverage when compared to `Tweettokenizer`)

- **GPT2Tokenizer**: 
   - Uses pre-trained GPT-2 vocabulary 

#### Data Preparation

Before training, prepare the dataset and build the vocabulary:
NOTE: This step assumes that you already have the files, `training.1600000.processed.noemoticon.csv` &
`testdata.manual.2009.06.14.csv` under the __same exact name__ inside `data/trainingandtestdata/`

```bash

# Prepare data and build vocabulary for desired tokenizer
python prepare.py --tokenizer tweet    # TweetTokenizer
python prepare.py --tokenizer bpe      # BPETokenizer  
python prepare.py --tokenizer gpt2     # GPT2Tokenizer
```

#### Start Training
To initiate a training pipeline with minimal configuration:

```bash
python training/train.py
```

**Training with custom options**
```bash
python training/train.py \
    --data_path /path/to/custom/data.csv \
    --save_freq 10 \
    --wandb
```

**Resume training from checkpoint:**
```bash
python training/train.py --resume training/weights/transformer/bpe/run_12345/15.pth
```

**Note**: All training parameters (model, tokenizer, epochs, batch_size, learning_rate, etc.) are configured in `config.py`. The CLI only accepts essential runtime arguments:
- `--data_path`: Training data file path (default: from config)
- `--save_freq`: Save frequency in epochs (default: 5)
- `--resume`: Checkpoint path to resume from
- `--wandb`: Enable Weights & Biases logging

#### Add-ons

- **Cosine Learning Rate Decay**: Automatic learning rate scheduling with warmup
- **Early Stopping**: Prevents overfitting with configurable patience
- **Model Checkpointing**: Automatic saving of best models
- **Metrics Tracking**: Accuracy, F1, Precision, Recall, Confusion matrix, Classification report
- **Mixed Precision**: Optimized training with torch.compile

### Post-Training Evaluation

1. After training, evaluate your model's performance on the test dataset:

```bash
python training/evaluate.py --model_path training/weights/transformer/bpe/run_12345/9.pth
```

**Note**: Model architecture and tokenizer are automatically inferred from the checkpoint metadata. No additional parameters needed.

2. Choose the best performing model based on validation metrics
```bash
cp training/weights/transformer/bpe/run_12345/best_model.pth training/inference/best.pth
```

3. **Update Configuration**: Modify `config.py` to point to your deployed model:
```python
# App Configs in config.py
MODEL_NAME = "transformer"
MODEL_PATH = "training/inference/best_model.pth"
TOKENIZER = "bpe"  # or "gpt2", "tweet"
```
<br>

## Running the Application

#### Model Configuration
NOTE: Before starting the FastAPI server, ensure the model parameters in `config.py` are correctly set:
You can download the pretrained weights from [here](https://drive.google.com/drive/folders/1ckwXgusr7Ygo-BRiie0j6HCV6cC6W69O?usp=sharing) and place it inside `training/inference/`
#### Start the Server
From the root, start the server:
```bash
uvicorn main:app --reload --port 8080
```
App should be running at `http://localhost:8080/`

#### API Endpoints and Usage
1. **Health Check Endpoint**
```bash
curl -X GET http://localhost:8080/
```
Response: Simple status confirmation

2. **Sentiment Analysis Endpoint**
```bash
curl -X POST http://localhost:8080/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I love this amazing product! It works perfectly."
  }'
```
**Example Response:**
```json
{
  "text": "love amazing product works perfectly",
  "prediction": "positive", 
  "confidence": 0.87
}
```

#### Response Schema
The API returns a structured response with the following fields:

- **`text`** (string): The cleaned and preprocessed version of your input text
  - Removes URLs, mentions, and other characters, refer: [code](./data/utils.py)
  - Converts to lowercase and applies tokenization preprocessing

- **`prediction`** (string): The predicted sentiment class
  - **"positive"**: Text expresses positive sentiment
  - **"negative"**: Text expresses negative sentiment

- **`confidence`** (float): Prediction confidence score between 0.0 and 1.0
  - Calculated using `softmax` probabilities from the model output
<br>

## Results
Performance evaluated on hold-out (`test.csv`) data
NOTE: The `NEUTRAL` category in `test.csv` is _disregarded_ for evaluation. The task is considered 
as a `binary` classification problem

### Model Architecture

The following results were obtained using a **Transformer** model with the architecture below, trained for **50 epochs** with **batch size 128**:

```
Transformer(
  (embedding): Embedding(30000, 128, padding_idx=0)
  (pos_encoding): PositionalEncoding(
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (transformer): ModuleList(
    (0-2): 3 x TransformerBlock(
      (attn): MultiHeadAttention(
        (q_linear): Linear(in_features=128, out_features=128, bias=True)
        (k_linear): Linear(in_features=128, out_features=128, bias=True)
        (v_linear): Linear(in_features=128, out_features=128, bias=True)
        (out_linear): Linear(in_features=128, out_features=128, bias=True)
      )
      (ln1): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (ln2): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
      (ff): FeedForward(
        (linear1): Linear(in_features=128, out_features=512, bias=True)
        (linear2): Linear(in_features=512, out_features=128, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
        (act): GELU(approximate='none')
      )
      (dropout): Dropout(p=0.1, inplace=False)
    )
  )
  (dropout): Dropout(p=0.1, inplace=False)
  (classifier): Linear(in_features=128, out_features=2, bias=True)
)
```
<br>

### Tokenizer Comparison Summary

| Tokenizer | Accuracy | Precision | Recall | F1 Score |
|-----------|----------|-----------|--------|----------|
| **GPT-2** | **0.8384** | **0.8100** | **0.8901** | **0.8482** |
| **BPE** | 0.8217 | **0.7864** | **0.8901** | **0.8351** |
| **Tweet** | 0.8189 | 0.7970 | 0.8626 | 0.8285 |

**Key Findings:**
- **GPT-2 tokenizer** achieves the best overall performance with highest accuracy (83.01%) and F1 score (83.82%)
- **BPE tokenizer** shows competitive performance with best precision (80.41%)
---

### 1. **BPE Tokenizer** with **Transformer** 

#### Overall Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 0.8217 |
| **F1 Score** | 0.8351 |
| **Precision** | 0.7864 |
| **Recall** | 0.8901 |

#### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Negative (0)** | 0.87 | 0.75 | 0.81 | 177 |
| **Positive (1)** | 0.79 | 0.89 | 0.84 | 182 |
| | | | | |
| **Accuracy** | | | 0.82 | 359 |

#### Confusion Matrix

|  | Predicted Negative | Predicted Positive |
|--|-------------------|-------------------|
| **Actual Negative** | 133 | 44 |
| **Actual Positive** | 20 | 162 |

### 2. **GPT-2 Tokenizer** with **Transformer**

#### Overall Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 0.8384 |
| **F1 Score** | 0.8482 |
| **Precision** | 0.8100 |
| **Recall** | 0.8901 |

#### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Negative (0)** | 0.87 | 0.79 | 0.83 | 177 |
| **Positive (1)** | 0.81 | 0.89 | 0.85 | 182 |
| | | | | |
| **Accuracy** | | | 0.83 | 359 |

#### Confusion Matrix

|  | Predicted Negative | Predicted Positive |
|--|-------------------|-------------------|
| **Actual Negative** | 139 | 38 |
| **Actual Positive** | 20 | 162 |

### 3. **Tweet Tokenizer** with **Transformer**

#### Overall Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 0.8189 |
| **F1 Score** | 0.8285 |
| **Precision** | 0.7970 |
| **Recall** | 0.8626 |

#### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Negative (0)** | 0.85 | 0.77 | 0.81 | 177 |
| **Positive (1)** | 0.80 | 0.86 | 0.83 | 182 |
| | | | | |
| **Accuracy** | | | 0.82 | 359 |

#### Confusion Matrix

|  | Predicted Negative | Predicted Positive |
|--|-------------------|-------------------|
| **Actual Negative** | 137 | 40 |
| **Actual Positive** | 25 | 157 |





