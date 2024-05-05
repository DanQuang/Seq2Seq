import datasets

dataset = datasets.load_dataset("bentrevett/multi30k")

print(type(dataset["train"]))