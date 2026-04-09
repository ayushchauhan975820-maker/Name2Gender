import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class GenderNaiveBayes:
    def __init__(self, alpha):
        self.priors = {}
        self.mle = {}
        self.mle[0] = {}
        self.mle[1] = {}
        self.alpha = alpha
        
        for i in [0, 1]:
            self.mle[i]["length"] = {}
            self.mle[i]["vowel_ratio"] = {}
            self.mle[i]["lst_three"] = {}

    def extract(self, name: str):
        info = {}
        lower_name = name.lower()
        length = len(lower_name)
        vowels = sum(1 for ch in lower_name if ch in 'aeiou')

        vowel_ratio = round(vowels / length, 2) if length > 0 else 0
        last_three = lower_name[-3:] if len(lower_name) >= 3 else lower_name

        info = {
            "length": len(lower_name),
            "vowel_ratio": vowel_ratio,
            "lst_three": last_three,
        }

        return info
    
    def count_tot(self, feature):
        if feature == "vowel_ratio": return 100
        if feature == "lst_three": return (26 * 26 * 26)
        return 50

    def fit(self, df: pd.DataFrame):
        tot = len(df)
        boys_count = len(df[df["Gender"] == 1])
        girls_count = len(df[df["Gender"] == 0])

        self.priors["boys_prior"] = (boys_count + self.alpha)/(tot + 2 * self.alpha)
        self.priors["girls_prior"] = (girls_count + self.alpha)/(tot + 2 * self.alpha)

        for name, Gender in zip(df["Name"], df["Gender"]):
            info = self.extract(name)

            for feature in info:
                self.mle[Gender][feature][info.get(feature)] = self.mle[Gender][feature].get(info.get(feature), 0) + 1

        for Gender in self.mle:
            for feature in self.mle[Gender]:
                for value in self.mle[Gender][feature]:
                    ct = girls_count if Gender == 0 else boys_count
                    prev = self.count_tot(feature) * self.alpha
                    self.mle[Gender][feature][value] = (self.mle[Gender][feature][value] + self.alpha)/(ct + prev)

        self.class_totals = {0: girls_count, 1: boys_count}

    def predict(self, df):
        result = []
        for name in df:
            info = self.extract(name)

            score_0 = np.log(self.priors.get("girls_prior"))
            score_1 = np.log(self.priors.get("boys_prior"))

            for feature in info:
                val = info[feature]
                prob_g = 0
                prob_b = 0
                prev = self.count_tot(feature)
                
                prob_g = self.mle[0][feature].get(val, self.alpha / (self.class_totals.get(0) + prev * self.alpha))
                prob_b = self.mle[1][feature].get(val, self.alpha / (self.class_totals.get(1) + prev * self.alpha))

                score_0 += np.log(prob_g)
                score_1 += np.log(prob_b)

            if(score_1 >= score_0): result.append(1)
            else: result.append(0)

        return pd.DataFrame(result)