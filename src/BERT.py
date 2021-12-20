import numpy as np
import pandas as pd
import torch, os
# this is the package for the light version of BERT
import transformers as ppb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

available_names = dict()
books_path = "/Users/leocheung/Desktop/brm_project/pure_cleaned_data/"


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
    excel_path = os.path.join("/Users/leocheung/Desktop/comp550_project/resources/author_info_revised.xlsx")
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
                        batch_number = len(lines)
                        randomRows = np.random.randint(batch_number, size=2000)
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

    # df = pd.read_csv("/Users/leocheung/Desktop/comp550_project/resources/sample.tsv", delimiter='\t', header=None)
    # df.head()
    batch_1 = df.sample(n=4000)
    df = None
    dataset = None
    # batch_1 = df
    # exit(0)
    # import the BERT model even though this is a distilled version (i.e. light version compared to the original BERT)
    model_class, tokenizer_class, pretrained_weights = (
        ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights)

    print("Start to tokenize the texts")
    tokenized = batch_1["text"].apply(
        (lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=512, truncation=True)))
    print("Finish to tokenize the texts")
    # exit(0)

    max_len = 0
    for i in tokenized.values:
        if len(i) > max_len:
            max_len = len(i)

    padded = np.array([i + [0] * (max_len - len(i)) for i in tokenized.values])

    attention_mask = np.where(padded != 0, 1, 0)

    print("Start to encode the texts")
    input_ids = torch.tensor(padded)
    attention_mask = torch.tensor(attention_mask)

    print("Start to train!")
    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    features = last_hidden_states[0][:, 0, :].numpy()
    print("Finish to encode the texts")

    labels = batch_1["label"]

    print("Start to split the dataset")
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels)

    # parameters = {'C': np.linspace(0.0001, 100, 20, 1)}
    # grid_search = GridSearchCV(LogisticRegression(max_iter=100), parameters)
    # grid_search.fit(train_features, train_labels)
    #
    # print('best parameters: ', grid_search.best_params_)
    # print('best scrores: ', grid_search.best_score_)

    lr_clf = LogisticRegression(max_iter=1000)
    print("Start to train the logistic regression models")
    lr_clf.fit(train_features, train_labels)
    print("The accuracy is " + str(lr_clf.score(test_features, test_labels)))


if __name__ == "__main__":
    main()
