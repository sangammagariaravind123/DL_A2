# Latin-to-Devanagari Transliteration (Seq2Seq with RNNs)

This project implements a **character-level sequence-to-sequence (Seq2Seq)** model in **Keras** for transliterating words written in the **Latin script** (e.g., "namaste") into the **Devanagari script** (e.g., "नमस्ते").

---

## Model Architecture

The model is a classic **encoder-decoder RNN** architecture, built using Keras:

1. **Input Embedding Layer**
   - Character embeddings for the Latin input sequence.

2. **Encoder RNN**
   - A single-layer **GRU** (can be swapped with LSTM).
   - Encodes the input Latin character sequence into a fixed-length context vector.

3. **Decoder RNN**
   - A GRU that takes the encoder’s final state as its initial state.
   - Predicts one Devanagari character at a time using **teacher forcing** during training.

4. **Dense Layer**
   - Applies a softmax over the Devanagari vocabulary to get probabilities at each time step.

---

##  Dataset

- Consists of paired sequences: Latin input and corresponding Devanagari target.
- Example: Latin: namaste Devanagari: नमस्ते

- Each word is tokenized into characters, then mapped to integer indices.

---

##  Training

- Loss: `SparseCategoricalCrossentropy`
- Optimizer: `Adam`
- Teacher Forcing used during training
- Outputs shape: `(batch_size, target_seq_len, vocab_size)`

---

##  Prediction

- Model predicts a sequence of indices.
- You can convert predictions to text using a `index_to_char` dictionary.
- Example predicted output (indices): [0, 27, 43, 35, 42, 1, 29, 30, ...]
  Converts to: नमस्ते


---



# Fine tuning GPT-2 to Generate English Lyrics
This project fine-tunes the GPT-2 language model on a collection of English song lyrics. After training, the model can generate new, creative song lines in a similar style.

##  What’s Included
- `cleaned_lyrics.txt`: A text file containing thousands of English song lyrics.
- Python code to fine-tune GPT-2 on these lyrics.
- Instructions to train and test the model.

## How It Works
1. The lyrics are cleaned and loaded.
2. We use a GPT-2 model from Hugging Face Transformers.
3. The model is trained using the lyrics as input.
4. After training, the model can generate lyrics from a starting line.

##  Requirements
- Python
- Transformers library (`pip install transformers`)
- Datasets (`pip install datasets`)
- (Optional) wandb for tracking training (`pip install wandb`)

##  Run Training
```python
# Load tokenizer and model
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Tokenize your dataset
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

#  Generate Lyrics
from transformers import pipeline

generator = pipeline("text-generation", model="./results", tokenizer="gpt2")
print(generator("I got a feeling", max_length=100)[0]["generated_text"])