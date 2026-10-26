# Next Word Predictor (LSTM) — Trained on Kafka's *Metamorphosis*

A next-word prediction and text generation model built with **TensorFlow/Keras**. It is trained on Franz Kafka's *The Metamorphosis* (Project Gutenberg) and uses an **Embedding + LSTM** network to predict the next word in a sentence, then chains predictions together to generate new text.

## Example

**Seed text:** `Gregor woke up and`
**Words generated:** 30

```
Gregor woke up and had to himself the chief clerk s firm footsteps in his highly
polished boots running while job would have to rush like now she would have a loud
filled it
```

The output picks up Kafka's vocabulary and phrasing, but it is not grammatically coherent. See [Limitations](#limitations).

## How It Works

1. **Load data** – Downloads the book text from Project Gutenberg using `tf.keras.utils.get_file`.
2. **Clean text** – Strips the Gutenberg header/footer, removes `\r`, lowercases, and splits into lines.
3. **Tokenize** – Keras `Tokenizer` with a custom punctuation filter builds the vocabulary (2,643 unique words).
4. **Build n-gram sequences** – For each line, every prefix becomes a training sample (e.g. `[53, 151]`, `[53, 151, 57]`, ...), giving **20,550 sequences**.
5. **Pad** – Sequences are pre-padded to the max length (18).
6. **Split inputs/labels** – `X` = all tokens except the last, `y` = the last token (one-hot encoded over 2,644 classes).
7. **Train** – Embedding → LSTM → Softmax classifier.
8. **Generate** – Repeatedly predict the most likely next word (`argmax`) and append it to the seed text.

## Model Architecture

| Layer | Output Shape | Params |
|---|---|---|
| Input | (None, 17) | 0 |
| Embedding (dim = 100) | (None, 17, 100) | 264,400 |
| LSTM (150 units) | (None, 150) | 150,600 |
| Dense (softmax, 2644) | (None, 2644) | 399,244 |
| **Total** | | **814,244** |

- **Loss:** categorical cross-entropy
- **Optimizer:** Adam
- **Metrics:** accuracy, precision, recall, AUC
- **Callbacks:** `EarlyStopping`, `ModelCheckpoint` (saves `best_kafka_predictor.keras`), `ReduceLROnPlateau`

## Results

After 100 epochs, on the **training data**:

| Metric | Value |
|---|---|
| Accuracy | ~92.5% |
| Precision | ~0.97 |
| Recall | ~0.91 |
| AUC | ~0.9999 |
| Loss | ~0.23 |

> These are training metrics only (no validation split). See the note below.

## Getting Started

### Requirements

- Python 3.9+
- TensorFlow 2.x
- NumPy

```bash
pip install tensorflow numpy
```

### Run

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
jupyter notebook Next_Word_Predictor.ipynb
```

Run all cells in order. The dataset downloads automatically, so an internet connection is needed on the first run.

### Generate your own text

Edit the last cell of the notebook:

```python
user_starting_text = "Gregor woke up and"
user_word_count = 30

final_story = generate_kafka_text(
    seed_text=user_starting_text,
    next_words=user_word_count,
    model=model,
    tokenizer=tokenizer,
    max_sequence_len=18
)
print(final_story)
```

## Project Structure

```
.
├── Next_Word_Predictor.ipynb   # Full pipeline: data prep, training, generation
├── best_kafka_predictor.keras  # Saved model checkpoint (created after training)
└── README.md
```

## Limitations

- **Overfitting:** The model is trained and evaluated on the same data (no validation set), so the ~92% accuracy reflects memorization of a small book rather than real generalization.
- **Inactive callbacks:** `EarlyStopping`, `ModelCheckpoint`, and `ReduceLROnPlateau` monitor `val_accuracy` / `val_loss`, which don't exist without validation data. Keras warns about this, and the checkpoint saves every epoch instead of only the best one.
- **Greedy decoding:** Always choosing the top word (`argmax`) can produce repetitive or odd text. Sampling with temperature would give more varied output.
- **Small corpus:** One short book (~120k characters) limits the vocabulary and quality of generated text.
- **Hard-coded sizes:** The vocabulary size (2644) and sequence length (17/18) are fixed in the code and should be derived from `total_words` and `max_sequence_len` if the dataset changes.

## Ideas for Improvement

- Add `validation_split=0.2` in `model.fit` so the callbacks work properly
- Add `Dropout` (already imported) and/or stack another LSTM or Bidirectional LSTM layer
- Use temperature or top-k sampling for generation
- Try pretrained embeddings (GloVe, Word2Vec)
- Train on a larger corpus, or move to a Transformer-based model

## Data Source

Text from [Project Gutenberg](https://www.gutenberg.org/ebooks/5200) — *Metamorphosis* by Franz Kafka (public domain).
