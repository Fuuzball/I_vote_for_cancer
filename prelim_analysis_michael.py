import os
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

train_variants_df = pd.read_csv("./data/training_variants")
test_variants_df = pd.read_csv("./data/test_variants")
train_text_df = pd.read_csv("./data/training_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])
test_text_df = pd.read_csv("./data/test_text", sep="\|\|", engine="python", skiprows=1, names=["ID", "Text"])
print("Train Variant".ljust(15), train_variants_df.shape)
print("Train Text".ljust(15), train_text_df.shape)
print("Test Variant".ljust(15), test_variants_df.shape)
print("Test Text".ljust(15), test_text_df.shape)
