# Seq2Seq
This is the Seq2Seq model for the Machine Translation task, using Pytorch

## About
Use LSTM as encoder to map input sentence to vector and another LSTM to decode target sentence from the vector. This Example use `bentrevett/multi30k` from [datasets](https://pypi.org/project/datasets/)

## Usage
To train the Seq2Seq model, use the command line:

``bash
python main.py --config config.yaml
``

To test the Seq2Seq model, go to main.py and command these codes:

```python
logging.info("Training started...")
Train_Task(config).train()
logging.info("Train complete")
```
