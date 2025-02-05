# Setup

# Python version

Install Python 3.12.3

## Dependencies

pip install torch torchvision transformers datasets torchaudio
pip install git+https://github.com/openai/CLIP.git
pip install opencv-python
pip install ftfy regex tqdm

# Getting Started

1. Create the following directories: dataset dataset/image tmp tmp/screenshot
2. Run the import: `python import.py`
3. Run the model fine tunning: `python fine_tuning.py`
4. Run the recognizer: `python recognizer.py`

You are ready to go.

# Test

In order to test the performance of model the folder `test` has some test data.

The test data is organized in the following way:

Screenshot-{n}.png --> That is a screenshot of MTG arena with multiple cards to be recognized
Screenshot-{n}-answers.txt --> The list of card names in the sequence displayed in the screenshot
