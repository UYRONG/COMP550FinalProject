import pandas as pd
import random
import numpy
import os
# this is the package for the light version of BERT
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords

stop = list(stopwords.words('english'))

available_names = dict()
books_path = "/Users/yuqiliu/Desktop/550/grouped_data"


# if you would like to include the stylo features
# you should add them
def get_stylo_features(text, num_sentences):
    # in this project, we only consider 5 lexical features
    # 1. average sentence length in characters
    # 2. average sentence length in words
    # 3. average word length
    # 4. total number of word token
    # 5. total number of word type
    # 6. num of nouns
    # 7. num of adjectives
    # 8. num of verbs
    original = text
    text = str(text).replace(".", " ").replace(",", " ").lower()
    feature_1 = len(text) / num_sentences
    words = text.split()
    feature_2 = len(words) / num_sentences
    feature_3 = sum(len(word) for word in words) / (len(words) + 1)
    feature_4 = len(words)
    feature_5 = len(list(set(words)))
    pos_tags = []
    # print(pos_tags)
    nouns = 0
    adjectives = 0
    verbs = 0
    for pair in pos_tags:
        if str(pair[1]).startswith("NN"):
            nouns += 1

        if str(pair[1]).startswith("JJ"):
            adjectives += 1

        if str(pair[1]).startswith("VB"):
            verbs += 1
    print([feature_1, feature_2, feature_3, feature_4, feature_5, nouns, adjectives, verbs])
    return [feature_1, feature_2, feature_3, feature_4, feature_5, nouns, adjectives, verbs]


def get_stylo_matrix(texts, num_sen):
    matrix = []
    for text in texts:
        matrix.append(get_stylo_features(text, num_sen))

    return numpy.array(matrix)


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

            '''if row["Genre"] != "SciFi":
                continue'''

            author_dic[row["Name"]] = Author(row["Name"], row["Gender"],
                                             row["Genre"], int(row["Date of Birth"]), row["Country of Birth"])
        except ValueError:
            print("Some error occurred for author " + row["Name"] + " in the excel file.")

    return author_dic


def main():
    # do some author retrieving stuff
    preload_names()
    author_dic = load_author_info()
    print("There are currently " + str(len(author_dic.keys())) + " available authors")
    dataset = []
    for author_name in author_dic.keys():
        correct_name = retrieve_correct_name(author_name)
        if correct_name is not None:
            datum = []
            with os.scandir(books_path + "/" + author_name) as author_dir:
                for book in author_dir:
                    if book.is_file() and not book.name.startswith('.') and (
                            book.name.endswith('txt') or book.name.endswith('TXT')):
                        file = open(os.path.abspath(book), 'r', encoding="utf-8")
                        selected_lines = []
                        lines = file.read().splitlines()
                        batch_number = len(lines) // 20
                        randomRows = numpy.random.randint(batch_number, size=2000)
                        for i in randomRows:
                            selected_lines = selected_lines + lines[i * 20:i * 20 + 20]

                        text = " ".join(selected_lines).replace("\n", " ")

                        if author_dic[author_name].birth >= 1945:
                            label = 1

                        if author_dic[author_name].birth < 1945:
                            label = 0
                        datum = [text, label]
                        dataset.append(datum)
                        # break
            # dataset.append(datum)
    # print(dataset)
    df = pd.DataFrame(dataset, columns=['text', 'label'])
    # create balanced labeled dataset
    balance_size = min(df.groupby('label').size()[0], df.groupby('label').size()[1])
    df = (df.groupby('label', as_index=False)
          .apply(lambda x: x.sample(n=balance_size))
          .reset_index(drop=True))

    batch_1 = df
    df = None
    dataset = None
    train_features, test_features, train_labels, test_labels = train_test_split(batch_1["text"], batch_1["label"],
                                                                                train_size=0.85)
    print("Finished splitting the data")
    # Create feature vectors
    vectorizer = TfidfVectorizer(stop_words=set(stop),
                                 ngram_range=(2, 2),
                                 sublinear_tf=True,
                                 use_idf=True)
    print("start to vectorize the texts")
    train_vectors = vectorizer.fit_transform(list(train_features))
    test_vectors = vectorizer.transform(list(test_features))
    # stylo_features = get_stylo_matrix(list(train_features), 10000)
    print("finish to vectorize the texts")
    # Perform classification with SVM, kernel=linear
    classifier = MultinomialNB()
    print("start to train the texts")
    # train_vectors = numpy.concatenate((train_vectors.toarray(), stylo_features), axis=1)
    classifier.fit(train_vectors, list(train_labels))
    # stylo_features = get_stylo_matrix(list(test_features), 10000)
    # test_vectors = numpy.concatenate((test_vectors.toarray(), stylo_features), axis=1)
    print("finish to train the texts")
    print(classifier.score(test_vectors, test_labels))


if __name__ == "__main__":
    main()
