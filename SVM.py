import pandas as pd
import numpy
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from nltk.corpus import stopwords
stop = list(stopwords.words('english'))


available_names = dict()
books_path = "/Users/yuqiliu/Desktop/550/grouped_data"
numpy.random.seed(0)

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
                        batch_number = len(lines)//20
                        randomRows = numpy.random.randint(batch_number, size=500)
                        for i in randomRows:
                            selected_lines = selected_lines + lines[i*20:i*20+20]                    
                        
                        text = " ".join(selected_lines).replace("\n", " ")

                        if author_dic[author_name].birth >= 1945:
                            label = 1

                        if author_dic[author_name].birth < 1945:
                            label = 0
                        datum = [text, label]
                        dataset.append(datum)
                        # break
            # dataset.append(datum)
    
    df = pd.DataFrame(dataset, columns=['text', 'label'])
    
    '''#create balanced labeled dataset
    balance_size = min(df.groupby('label').size()[0], df.groupby('label').size()[1])
    df = (df.groupby('label', as_index=False)
          .apply(lambda x: x.sample(n=balance_size))
        .reset_index(drop=True))'''

    batch_1 = df
    df = None
    dataset = None
    train_features, test_features, train_labels, test_labels = train_test_split(batch_1["text"], batch_1["label"], train_size=0.85)
    print("Finished splitting the data")
    # Create feature vectors
    vectorizer = TfidfVectorizer(
                                stop_words=set(stop),
                                 ngram_range=(1,2),
                                 sublinear_tf=True,
                                 use_idf=True)
    print("start to vectorize the texts")
    train_vectors = vectorizer.fit_transform(list(train_features))
    test_vectors = vectorizer.transform(list(test_features))
    print("finish to vectorize the texts")
    # Perform classification with SVM, kernel=linear
    classifier_linear = svm.SVC(kernel='linear')
    print("start to train the texts")
    classifier_linear.fit(train_vectors, list(train_labels))
    print("finish to train the texts")
  
    print(classifier_linear.score(test_vectors, test_labels))
   


if __name__ == "__main__":
    main()