markdown
Copy code
# German to English Translation using Transformers

This repository contains code to train a transformer model for translating German sentences to English using PyTorch and provides a Streamlit web application for inference.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/german-to-english-translation.git
   cd german-to-english-translation

##Install the required packages:

```bash
Copy code
pip install -r requirements.txt
##Download the necessary spaCy language models:

```bash
Copy code
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
##Training the Model
To train the model, run:

```bash
Copy code
python train.py

##Ensure you have the dataset in the data/ directory in the appropriate format. The script will preprocess the data, train the model, and save the trained model and vocabulary.

##Running the Streamlit App
To start the Streamlit app, run:

```bash
Copy code
streamlit run translate_app.py

##This will open a web browser with the translation interface where you can input German sentences and get English translations.

##File Descriptions
train.py: Script to preprocess data, train the transformer model, and save the trained model.
translate_app.py: Streamlit app script for performing translation inference.
vocab.pt: Saved vocabulary file.
transformer-model.pt: Saved trained model file.
data/: Directory containing the dataset files.
src/utils.py: Additional utility functions if needed.


##Directory Structure

```css
Copy code
german-to-english-translation/
│
├── README.md
├── requirements.txt
├── train.py
├── translate_app.py
├── vocab.pt
├── transformer-model.pt
├── data/
│   ├── train.de
│   ├── train.en
│   ├── valid.de
│   ├── valid.en
│   └── test.de
│   └── test.en
└── src/
    ├── __init__.py
    └── utils.py

##License
This project is licensed under the MIT License. See the LICENSE file for details.

##Acknowledgements
PyTorch
torchtext
spaCy
Streamlit
Dataset: Multi30k
