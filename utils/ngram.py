import pandas as pd
import os
import numpy
from collections import defaultdict
from sklearn.model_selection import train_test_split
import string

string.punctuation
import matplotlib.pyplot as plt

plt.style.use(style='seaborn')

available_names = dict()
stopwords = []
books_path = "/Users/yuqiliu/Desktop/550/pure_cleaned_data"
stopwaords_path = "/Users/yuqiliu/Desktop/550/stopwords.txt"
stopwords2 = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out',
              'very', 'having', 'with',
              'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself',
              'other', 'off', 'is', 's',
              'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we',
              'these', 'your', 'his', 'through',
              'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while',
              'above', 'both', 'up', 'to',
              'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have',
              'in', 'will', 'on', 'does',
              'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under',
              'he', 'you', 'herself', 'has',
              'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
              'if', 'theirs', 'my', 'against',
              'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than', 'i']


# a simple class to organize the information of each author
class Author:
    def __init__(self, name, gender, genre, birth, country):
        # the name of the author, string
        self.name = name
        # the gender of the author, string type, either F or M
        self.gender = gender
        # the genre of the author, string
        self.genre = genre
        # the year of the birth, int
        self.birth = birth
        # the country of birth, string
        self.country = country

    # for debug purpose mainly
    def __str__(self):
        return "Author Name: " + str(self.name) + "\n" + \
               "Gender: " + str(self.gender) + "\n" + \
               "Genre: " + str(self.genre) + "\n" + \
               "Birth: " + str(self.birth) + "\n" + \
               "Country of Birth: " + str(self.country) + "\n"


# pre-load all the authors names in the author directory to reduce
# the overhead for checking the existence through drive everytime
def preload_names():
    global available_names
    with os.scandir(books_path) as root:
        for author_name in root:
            if not author_name.is_file() and not author_name.name.startswith('.'):
                available_names[author_name.name.lower()] = author_name.name


# this functions is used to handle the case where names in excel file
# and the directory names only differ in upper and lower case
def retrieve_correct_name(intended_name):
    if intended_name.lower() in available_names.keys():
        return available_names[intended_name.lower()]
    else:
        return None


# read the author_info excel file
# convert each author into an author object
def load_author_info():
    excel_path = os.path.join("/Users/yuqiliu/Desktop/550/data.xlsx")
    data = pd.read_excel(excel_path, sheet_name='Sheet1')
    # convert it into dictionary
    df = pd.DataFrame(data, columns=['Name', 'Gender', 'Genre', 'Date of Birth', 'Country of Birth'])
    author_dic = dict()
    for index, row in df.iterrows():
        # remove extra empty space in the last character for some cases
        try:
            if row["Name"][-1] == " ":
                row["Name"] = row["Name"][:-1]

            if row["Country of Birth"] != "USA":
                continue

            if int(row["Date of Birth"]) < 1000:
                continue

            author_dic[row["Name"]] = Author(row["Name"], row["Gender"],
                                             row["Genre"], int(row["Date of Birth"]), row["Country of Birth"])
        except ValueError:
            print("Some error occurred for author " + row["Name"] + " in the excel file.")

    return author_dic


# defining the function to remove punctuation
def remove_punctuation(text):
    if (type(text) == float):
        return text
    if (text is None):
        return text
    ans = ""
    for i in text:
        if i not in string.punctuation:
            ans += i
    return ans


def generate_N_grams(text, ngram=1):
    text = text.lower()
    words = [word for word in text.split(" ") if word not in set(stopwords2)]
    # words=[word for word in text.split(" ")]
    words = list(filter(None, words))
    # print("Sentence after removing stopwords:",words)
    temp = zip(*[words[i:] for i in range(0, ngram)])
    ans = [' '.join(ngram) for ngram in temp]
    return ans


def main():
    global stopwords
    with open(stopwaords_path) as file:
        for line in file:
            line = line.strip()  # or some other preprocessing
            stopwords.append(line)

    preload_names()
    author_dic = load_author_info()
    dataset = []

    for author_name in author_dic.keys():
        correct_name = retrieve_correct_name(author_name)
        if correct_name is not None:
            datum = []
            with os.scandir(books_path + '/' + author_name) as author_dir:
                for book in author_dir:
                    if book.is_file() and not book.name.startswith('.') and (
                            book.name.endswith('txt') or book.name.endswith('TXT')):
                        file = open(os.path.abspath(book), 'r', encoding="utf-8")
                        selected_lines = file.readlines()[0:1000]
                        text = " ".join(selected_lines).replace("\n", " ")

                        if author_dic[author_name].birth >= 1945:
                            label = 'after'

                        if author_dic[author_name].birth < 1945:
                            label = 'before'
                        datum = [text, label]
                        dataset.append(datum)
                        # break
            dataset.append(datum)

    df = pd.DataFrame(dataset, columns=['text', 'label'])

    (x_train, x_test, y_train, y_test) = train_test_split(df["text"], df["label"], test_size=0.4)

    df1 = pd.DataFrame(x_train)
    df1 = df1.rename(columns={0: 'text'})
    df2 = pd.DataFrame(y_train)
    df2 = df2.rename(columns={0: 'label'})
    df_train = pd.concat([df1, df2], axis=1)

    df3 = pd.DataFrame(x_test)
    df3 = df3.rename(columns={0: 'text'})
    df4 = pd.DataFrame(y_test)
    df4 = df2.rename(columns={0: 'label'})
    df_test = pd.concat([df3, df4], axis=1)

    # storing the puntuation free text in a new column called clean_msg
    df_train['text'] = df_train['text'].apply(lambda x: remove_punctuation(x))
    df_test['text'] = df_test['text'].apply(lambda x: remove_punctuation(x))

    # examples of unigram
    positiveValues = defaultdict(int)
    negativeValues = defaultdict(int)

    # get the count of every word in both the columns of df_train and df_test dataframes where sentiment="positive"
    for text in df_train[df_train.label == "after"].text:
        for word in generate_N_grams(text, 3):
            positiveValues[word] += 1

    # get the count of every word in both the columns of df_train and df_test dataframes where sentiment="negative"
    for text in df_train[df_train.label == "before"].text:
        for word in generate_N_grams(text, 3):
            negativeValues[word] += 1

    df_positive = pd.DataFrame(sorted(positiveValues.items(), key=lambda x: x[1], reverse=True))
    df_negative = pd.DataFrame(sorted(negativeValues.items(), key=lambda x: x[1], reverse=True))
    pd1 = df_positive[0][:25]
    pd2 = df_positive[1][:25]
    ned1 = df_negative[0][:25]
    ned2 = df_negative[1][:25]

    print("*" * 50)
    print(pd1)
    print(pd2)
    print(ned1)
    print(ned2)


if __name__ == "__main__":
    main()
