import pandas as pd
import matplotlib.pyplot as plt

before_words = ["time", "little", "eyes", "looked", "head", "hand", "door", "people", "own", "look"]
before_counts = [75070, 47204, 44543, 43197, 36683, 36631, 35585, 33432, 33292, 32633]

after_words = ["time", "eyes", "looked", "head", "little", "hand", "people", "look", "hed", "door"]
after_counts = [80474, 60070, 53764, 49250, 44715, 43826, 38872, 38003, 37016, 33846]

plt.bar(before_words, before_counts, color='lightskyblue',
        width=0.5)
plt.xlabel("Word Types", fontsize=13)
plt.ylim(0, 83000)
plt.ylabel("Frequency", fontsize=13)
plt.title("Top 10 words before 1945 (Unigram)", fontsize=18)
plt.show()

plt.bar(after_words, after_counts, color='mediumaquamarine',
        width=0.5)
plt.xlabel("Word Types", fontsize=13)
plt.ylim(0, 83000)
plt.ylabel("Frequency", fontsize=13)
plt.title("Top 10 words after 1945 (Unigram)", fontsize=18)
plt.show()

# before_words2 = ["shook head", "closed eyes", "deep breath", "front door", "blue eyes", "leaned forward", "held hand",
#                  "little girl", "parking lot", "cleared throat"]
# before_counts2 = [7210, 1811, 1761]
#
# after_words2 = ["shook head", "closed eyes", "deep breath", "front door", "blue eyes", "leaned forward", "held hand",
#                  "little girl", "parking lot", "cleared throat"]
# after_counts2 = [10822, 2775, 2754, 2187, 2171, 1487, 1486, 1471, 1410, 1403]
