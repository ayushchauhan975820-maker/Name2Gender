import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class GenderNaiveBayes:
    def __init__(self, df):
        self.priors = {}
        self.mle = {}