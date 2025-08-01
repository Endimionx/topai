import numpy as np
from itertools import product
from collections import Counter

def score_combination(comb):
    score = 0
    digits = comb

    odd = sum(1 for d in digits if d % 2 == 1)
    big = sum(1 for d in digits if d >= 5)
    if 1 <= odd <= 3: score += 1
    if 1 <= big <= 3: score += 1
    if max(Counter(digits).values()) <= 2: score += 1

    if digits == sorted(digits) or digits == sorted(digits, reverse=True): score += 1
    elif digits[0] < digits[1] > digits[2] < digits[3] or digits[0] > digits[1] < digits[2] > digits[3]:
        score += 1

    return score

def filter_top_combinations(predictions, top_k=10):
    all_combinations = list(product(*predictions))
    scored = [(c, score_combination(c)) for c in all_combinations]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
