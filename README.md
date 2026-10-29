# Intent Detection Using Fine-Tuned DistilBERT

A Natural Language Processing (NLP) project that classifies user messages into predefined intents using a fine-tuned **DistilBERT** transformer model. The goal is to understand the purpose behind a user's message and identify the corresponding intent.

## Project Overview

Intent detection is an important component of conversational AI systems, virtual assistants, and chatbots. It helps a system determine what a user wants to accomplish based on their message.

In this project, I fine-tuned the pretrained `distilbert-base-uncased` model for a multi-class text classification task using the Hugging Face Transformers library.

The model is trained to recognize seven different intents, covering tasks such as playing music, checking weather, booking restaurants, and searching for creative works.

## Features

- Text preprocessing and normalization
- Intent label encoding
- Tokenization using the DistilBERT tokenizer
- Fine-tuning a pretrained transformer model
- Training and validation dataset preparation
- Model evaluation using a classification report and confusion matrix
- Saving the fine-tuned model and tokenizer
- Predicting the intent of new user messages

## Technologies Used

- **Python** — Core programming language
- **Pandas and NumPy** — Data manipulation
- **NLTK** — Stopword removal
- **Scikit-learn** — Dataset splitting and evaluation metrics
- **PyTorch** — Deep learning framework
- **Hugging Face Transformers** — DistilBERT model, tokenizer, and Trainer API
- **Hugging Face Datasets** — Dataset preparation and tokenization
- **Kaggle Notebooks** — Development and model training

## Dataset

The project uses the [Intent Recognition Dataset on Kaggle](https://www.kaggle.com/datasets/himanshunayal/intent-recognition-dataset).

The notebook loads the dataset from `train.csv` and uses the `text` and `intent` columns.

### Supported Intents

The model recognizes the following seven classes:

| Intent | Description |
|---|---|
| `PlayMusic` | Requests related to playing music |
| `AddToPlaylist` | Adding music to a playlist |
| `RateBook` | Rating a book |
| `SearchScreeningEvent` | Searching for screening events |
| `BookRestaurant` | Booking a restaurant |
| `GetWeather` | Asking about weather |
| `SearchCreativeWork` | Searching for creative works |

## Project Workflow

1. **Data loading:** Load the intent recognition dataset into a Pandas DataFrame.
2. **Text preprocessing:** Convert text to lowercase and remove English stopwords using NLTK.
3. **Label encoding:** Convert intent categories into numerical class IDs.
4. **Tokenization:** Use `DistilBertTokenizer` to tokenize the cleaned text and prepare model inputs.
5. **Dataset splitting:** Split the data into training and validation subsets using an 80:20 ratio with stratification.
6. **Model initialization:** Load the pretrained `distilbert-base-uncased` model with a classification head for seven intent classes.
7. **Fine-tuning:** Train the model using the Hugging Face `Trainer` API.
8. **Evaluation:** Generate a classification report and confusion matrix.
9. **Model saving:** Save the fine-tuned model and tokenizer for future inference.
10. **Inference:** Pass new messages to the model to obtain predicted class IDs and corresponding intents.

## Model Configuration

The notebook uses the following training configuration:

| Parameter | Value |
|---|---|
| Pretrained model | `distilbert-base-uncased` |
| Model type | DistilBERT for sequence classification |
| Number of classes | 7 |
| Training-validation split | 80:20 |
| Random state | 42 |
| Learning rate | `2e-5` |
| Training batch size | 16 |
| Evaluation batch size | 64 |
| Number of epochs | 3 |
| Weight decay | 0.01 |
| Maximum tokenized sequence length | 28 tokens |

## Evaluation

The notebook generates the following evaluation outputs:

- **Precision:** Measures how many predictions for an intent are correct.
- **Recall:** Measures how many examples of an intent the model correctly identifies.
- **F1-score:** The harmonic mean of precision and recall.
- **Confusion matrix:** Shows correct predictions and misclassifications across intent classes.

The notebook's current classification report displays approximately **99% accuracy**. However, this result was calculated using the full dataset rather than only the held-out validation subset, so it should not be interpreted as an unbiased estimate of generalization performance.

For a proper evaluation, predictions should be generated using `eval_dataset`, which contains examples excluded from training.

## Model Saving

The fine-tuned model and tokenizer are saved using Hugging Face's `save_pretrained()` method.

```python
model.save_pretrained("./saved_fine_tuned_model")
tokenizer.save_pretrained("./saved_fine_tuned_model")
```

This creates a directory containing the model weights, model configuration, and tokenizer files needed to reload the model.

**Note:** The notebook's saved model directory is a local training artifact. To make inference available to others, upload the required model files to a model repository or provide instructions for reproducing the training process.

## Example Use Cases

This type of intent detection system can be used as a building block for:

- Voice assistants
- Customer support chatbots
- Music recommendation and control systems
- Restaurant booking assistants
- Weather information bots
- Conversational search systems

## Limitations and Future Improvements

- Evaluate the model on a strictly held-out validation or test set.
- Preserve the exact training label-to-ID mapping during inference to ensure predictions are translated into the correct intent names.
- Experiment with different learning rates, batch sizes, and training epochs.
- Compare performance with other pretrained transformer models.
- Explore deployment through FastAPI or Streamlit.
- Test the model on real-world messages that differ from the training data.

## What I Learned

Through this project, I practiced:

- Fine-tuning pretrained transformer models for text classification
- Working with DistilBERT and Hugging Face Transformers
- Preparing datasets for transformer-based NLP tasks
- Training models with PyTorch and the Hugging Face Trainer API
- Evaluating classification models using scikit-learn
- Saving and reloading trained models for inference

## Repository Structure

```text
intent-detection/
├── intent-detection.ipynb
├── README.md
└── saved_fine_tuned_model/   # Optional: saved model artifacts
```

The saved model directory is optional if you plan to train the model directly from the notebook. Large model files can be hosted separately using the Hugging Face Hub.

## Author

Developed as an NLP project to explore transformer fine-tuning and multi-class intent classification using DistilBERT.
