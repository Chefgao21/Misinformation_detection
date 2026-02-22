# Enhanced Misinformation Detection with RAG & LLM Explanations

Complete fake news detection system with RAG pipeline and chain-of-thought explanations.



## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start Redis
```bash
# Mac
brew install redis
brew services start redis

# Ubuntu
sudo apt-get install redis-server
sudo systemctl start redis-server

# Verify
redis-cli ping  
```

### 3. Prepare Data
You need:
- `train.tsv`, `valid.tsv`, `test.tsv` (LIAR dataset)
- `fact_check_articles_averitec.csv` (AVeriTeC RAG articles)

### 4. Run Training
```bash
python debertav3_misinformation.py
python debertav3_langchain.py
```



## Architecture

### 1. RAG Pipeline
```
Take a user statement, convert it to a numerical vector (embedding), search Redis vector database, get the 3 most similar articles, combine those articles with the original statement
```

### 2. Model
```
Combine DeBERTa text embeddings and MLP-processed metadata features into a classifier that outputs the prediction
```

### 3. Explanations
```
Feed the statement, prediction, evidence, and speaker history into Llama 3.2 to generate a natural language explanation
```


## Files Structure

```
.
├── derbtav3_misinformation.py            # Main training script
├── derbtav3_langchain.py                 # Main training script with langchain
├── requirements.txt                      # Dependencies
├── README.md                             
├── train.tsv                             # Training data (LIAR)
├── valid.tsv                             # Validation data
├── test.tsv                              # Test data
├── fact_check_articles_averitec.csv      # fact checking articles (AVeriTeC)
├── rag.py                                # Preprocess AVeriTeC for RAG
├── rag.json                              # Postprocessed AVeriTeC 

```

## Configuration

Edit `Config` class in the script:

```python
class Config:
    MODEL_NAME = 'microsoft/deberta-v3-base'
    EXPLANATION_MODEL = 'meta-llama/Llama-3.2-3B-Instruct'
    EPOCHS = 5
    BATCH_SIZE = 16
    TOP_K_RETRIEVAL = 3  # Number of articles to retrieve
```


## Implementation Details

### RAG Retrieval
- Uses sentence-transformers/all-MiniLM-L6-v2 for embeddings
- Cosine similarity search in Redis
- Returns top-3 most similar fact-checking articles
- Context concatenated with statement before model input

### Explanation Generation
- Llama 3.2 3B Instruct for text generation
- Structured prompt with 5-step chain-of-thought
- Includes statement, speaker history, and retrieved evidence
- Generates 2-3 sentence explanations

### Data Flow
```
1. Load LIAR statements + AVeriTeC articles
2. Index articles in Redis (one-time)
3. For each statement:
   - Retrieve top-3 similar articles
   - Concatenate with statement
   - Feed to DeBERTa model
4. Generate explanations for sample predictions
5. Save results
```


## Citation

AVeriTeC Dataset:
```
@inproceedings{schlichtkrull-etal-2023-averitec,
    title = "AVeriTeC: A Dataset for Real-world Claim Verification",
    author = "Schlichtkrull, Michael and others",
    year = "2023"
}
```

## License

MIT License - Free to use and modify

## Acknowledgments

- Microsoft for DeBERTa
- Meta for Llama models
- Sentence-Transformers library
- Redis for vector storage
- AVeriTeC dataset creators
