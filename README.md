# Fine tuning GPT-2 to Generate English Lyrics
This project fine-tunes the GPT-2 language model on a collection of English song lyrics. After training, the model can generate new, creative song lines in a similar style.

##  What’s Included
- `cleaned_lyrics.txt`: A text file containing thousands of English song lyrics.
- Python code to fine-tune GPT-2 on these lyrics.
- Instructions to train and test the model.

## ⚙️ How It Works
1. The lyrics are cleaned and loaded.
2. We use a GPT-2 model from Hugging Face Transformers.
3. The model is trained using the lyrics as input.
4. After training, the model can generate lyrics from a starting line.

## ✅ Requirements
- Python
- Transformers library (`pip install transformers`)
- Datasets (`pip install datasets`)
- (Optional) wandb for tracking training (`pip install wandb`)
