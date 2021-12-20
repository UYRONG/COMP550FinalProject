# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import csv

from pandas import read_excel
import pandas as pd
import matplotlib.pyplot as plt

my_sheet = 'Sheet1'  # change it to your sheet name, you can find your sheet name at the bottom left of your excel file
file_name = 'usa_1900-1990.xlsx'  # change it to the name of your excel file
df = read_excel(file_name, sheet_name=my_sheet)
df = df.drop(columns='Unnamed: 5')
df = df.drop(columns='Unnamed: 6')
df = df.drop(columns='Unnamed: 7')
df = df.drop(columns='Country of Birth')
# df = df.loc[df['Country of Birth'] == 'USA']
df = df.loc[df['Date of Birth'] >= 1900]
df = df.loc[df['Date of Birth'] <= 1990]
df['BirthYearGroup'] = None
print(len(df))
df = df.apply(lambda x: x.astype(str).str.upper()).drop_duplicates(subset=['Name'], keep='first')
print(len(df))

df['Date of Birth'] = df['Date of Birth'].astype(int)

for i in range(len(df)):
    if df['Date of Birth'].values[i] <= 1945:
        df['BirthYearGroup'].values[i] = 0
    else:
        df['BirthYearGroup'].values[i] = 1

df['BirthYearGroup_new'] = None
for i in range(len(df)):
    group_value = int(df['Date of Birth'].values[i] / 10)
    df['BirthYearGroup_new'].values[i] = group_value * 10

print(df)
print(df['BirthYearGroup'].value_counts())
print(pd.DataFrame(df['BirthYearGroup_new'].value_counts()))
result = pd.DataFrame(df['BirthYearGroup_new'].value_counts())
new = df['Date of Birth'].to_list()
new_dict = {}
for i in range(1900, 1991):
    new_dict[i] = 0
for i in range(len(new)):
    new_dict[new[i]] += 1
print(new_dict)
courses = list(new_dict.keys())
values = list(new_dict.values())
#
# fig = plt.figure(figsize=(10, 5))

# creating the bar plot
plt.bar(courses, values, color='steelblue',
        width=0.5)

plt.xlabel("Birth Year", fontsize=16)
plt.ylabel("Number of Authors in Each Year", fontsize=16)
plt.title("The Birth Year of Author in USA in 1900s", fontsize=20)
plt.show()

print(pd.DataFrame(df['Genre'].value_counts()))
genre_key_value = df['Genre'].unique()
genre_dict = {i: 0 for i in genre_key_value}
print(genre_dict)
genre_list = df['Genre'].to_list()
for i in range(len(genre_list)):
    genre_dict[genre_list[i]] += 1
print(genre_dict)
labels = []
sizes = []

for x, y in genre_dict.items():
    labels.append(x)
    sizes.append(y)
colors = ['lightblue', 'plum', 'lightpink', 'lightsteelblue',
          'bisque', 'cornsilk', 'thistle', 'slategrey', 'khaki',
          'honeydew', 'pink', 'lavender', 'burlywood', 'yellow']
plt.pie(sizes, labels=labels, colors=colors)

plt.axis('equal')
# plt.title("Proportion of Genre")
plt.show()

filterinfDataframe = df[(df['Genre'] == 'SCIFI') & (df['Date of Birth'] <= 1945)]
filterinfDataframe2 = df[(df['Genre'] == 'SCIFI') & (df['Date of Birth'] > 1945) & (df['Date of Birth'] <= 1990)]
print("number of scifi author before 1945:" + str(len(filterinfDataframe)))
print("number of scifi author after 1945: " + str(len(filterinfDataframe2)))
#
# df['BirthYearGroup_new'].plot(kind='bar')
# plt.show()
# def main():
#     # Use a breakpoint in the code line below to debug your script.
#     print('aaa')  # Press ⌘F8 to toggle the breakpoint.
#
#
# # Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
