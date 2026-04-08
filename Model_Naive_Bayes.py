import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class GenderNaiveBayes:
    def __init__(self):
        self.priors = {}
        self.mle = {}
        self.mle[0] = {}
        self.mle[1] = {}
        
        for i in [0, 1]:
            self.mle[i]["last"] = {}
            self.mle[i]["first"] = {}
            self.mle[i]["length"] = {}
            self.mle[i]["vowel_count"] = {}

    def extract(self, name: str):
        info = {}
        lower_name = name.lower()
        info = {
            "last": lower_name[-1],
            "first": lower_name[0],
            "length": len(lower_name),
            "vowel_count": sum(1 for ch in lower_name if ch in 'aeiou')
        }

        return info
    
    def count_tot(self, feature):
        if feature == "last" or feature == "first": return  26
        if feature == "vowel_count": return 20
        return 50

    def fit(self, df: pd.DataFrame):
        tot = len(df)
        boys_count = len(df[df["Gender"] == 1])
        girls_count = len(df[df["Gender"] == 0])

        self.priors["boys_prior"] = (boys_count + 1)/(tot + 2)
        self.priors["girls_prior"] = (girls_count + 1)/(tot + 2)

        for name, Gender in zip(df["Name"], df["Gender"]):
            info = self.extract(name)

            for feature in info:
                self.mle[Gender][feature][info.get(feature)] = self.mle[Gender][feature].get(info.get(feature), 0) + 1

        for Gender in self.mle:
            for feature in self.mle[Gender]:
                for value in self.mle[Gender][feature]:
                    ct = girls_count if Gender == 0 else boys_count
                    prev = self.count_tot(feature)
                    self.mle[Gender][feature][value] = (self.mle[Gender][feature][value] + 1)/(ct + prev)

        self.class_totals = {0: girls_count, 1: boys_count}

    def predict(self, name: str):
        info = self.extract(name)

        score_0 = np.log(self.priors.get("girls_prior"))
        score_1 = np.log(self.priors.get("boys_prior"))

        for feature in info:
            val = info[feature]
            prob_g = 0
            prob_b = 0
            prev = self.count_tot(feature)
            
            prob_g = self.mle[0][feature].get(val, 1 / (self.class_totals.get(0) + prev))
            prob_b = self.mle[1][feature].get(val, 1 / (self.class_totals.get(1) + prev))

            score_0 += np.log(prob_g)
            score_1 += np.log(prob_b)

        if(score_1 >= score_0): return 1
        else: return 0