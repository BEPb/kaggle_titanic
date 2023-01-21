"""
Python 3.10 prediction program will you survive on the titanic or not
File name Titanic_gui.py

Version: 0.1
Author: Andrej Marinchenko
Date: 2022-04-21
"""

#  установка необходимых библиотек
import sys

from tkinter import *
import tkinter as tk
import tkinter.ttk as TTK
from PIL import ImageTk, Image
import random
import numpy as np

import pandas as pd
from sklearn.model_selection import train_test_split  # random split into training and test sets and Cross-validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# отдельно прописываем класс отображения информации в текстовом окне
class PrintLogger():  # create file like object
    def __init__(self, textbox):  # pass reference to text widget
        self.textbox = textbox  # keep ref

    def write(self, text):
        self.textbox.insert(tk.END, text)  # write text to textbox
            # could also scroll to end of textbox here to make sure always visible
            # также можно прокрутить до конца текстового поля здесь, чтобы убедиться, что он всегда виден

    def flush(self):  # needed for file like object
        pass

def get_text():
    Output.delete(1.0, END)
    abr_t = combobox_abr.get()
    name_t = Name.get()
    surname_t = SurName.get()
    pclass_t = combobox_tiket.get()
    sex_t = combobox_sex.get()
    age_t = Age_in_years.get()
    sibsp_t = combobox_sibsp.get()
    parch_t = combobox_parch.get()
    tiket_num_t = Ticket_number.get()
    fare_t = Passenger_fare.get()
    cabin_let_t = combobox_cabin_letter.get()
    cabin_t = Cabin_number.get()
    embarked_t = combobox_embarked.get()

    print(abr_t, ' ', name_t, ' ', surname_t)
    print(pclass_t)
    print(sex_t)
    print(age_t)
    print(sibsp_t)
    print(parch_t)
    print(tiket_num_t)
    print(fare_t)
    print(cabin_let_t, '-', cabin_t)
    print(embarked_t)

    # PassengerId, Pclass,      Name,          Sex,    Age,   SibSp,   Parch,     Ticket,  Fare, Cabin, Embarked
    # 892,            3,  "Kelly, Mr. James",  male,   34.5,    0,       0,       330911,  7.8292,      ,   Q
    # 918,            1,  "Ostby, Miss. Hel", female,  22,      0,       1,       113509,  61.9792, B36,    C

    PassengerId_t = random.randint(1, 2224)
    Pclass_t = pclass_t[0]
    Name_t = name_t + ', ' + abr_t + ' ' + surname_t
    Sex_t = sex_t
    Age_t = float(age_t)
    SibSp_t = sibsp_t[0]
    Parch_t = parch_t[0]
    Ticket_t = tiket_num_t
    Fare_t = fare_t
    Cabin_t = cabin_let_t + cabin_t
    Embarked_t = embarked_t[0]

    # Read in the training and test sets
    train_df = pd.read_csv('../data/train.csv')
    test_df = pd.DataFrame(np.array(
        [[PassengerId_t, Pclass_t, Name_t, Sex_t, Age_t, SibSp_t, Parch_t, Ticket_t, Fare_t, Cabin_t, Embarked_t],
         [PassengerId_t, Pclass_t, Name_t, Sex_t, Age_t, SibSp_t, Parch_t, Ticket_t, Fare_t, Cabin_t, Embarked_t]]),
        columns=['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin',
                 'Embarked'])

    ###################################### Preprocess the data #############################################################
    # Identify most relevant features
    # You can use techniques like feature importance or correlation analysis to help you identify the most important features
    relevant_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

    # Handle missing values
    imputer = SimpleImputer(strategy='most_frequent')
    train_df[relevant_features] = imputer.fit_transform(train_df[relevant_features])
    test_df[relevant_features] = imputer.transform(test_df[relevant_features])

    # Encode categorical variables as numeric
    train_df['Sex'] = train_df['Sex'].map({'male': 0, 'female': 1})
    test_df['Sex'] = test_df['Sex'].map({'male': 0, 'female': 1})
    train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    test_df['Embarked'] = test_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    # Transform skewed or non-normal features
    # Instead of normalizing all of the numeric features, you could try using techniques like log transformation or
    # Box-Cox transformation to make the distribution of a feature more normal
    scaler = StandardScaler()
    train_df[relevant_features] = scaler.fit_transform(train_df[relevant_features])
    test_df[relevant_features] = scaler.transform(test_df[relevant_features])

    # Split the data into features (X) and labels (y)
    X_train = train_df[relevant_features]
    y_train = train_df['Survived']
    X_test = test_df[relevant_features]

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2,
                                                      random_state=30)  # random split into training and test sets
    ############################################## Train the model #########################################################
    model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, max_depth=3, random_state=2))
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    if y_pred[0] == 0:
        print('You will be Not Survived!!!')
    else:
        print('You will be Survived!!!')
    # print(y_pred)
    # print(X_test)

root = tk.Tk()
root.title('Titanic predictor')  # титул окна
root.geometry('800x700')  # размер окна
root.maxsize(800, 700)
root.minsize(800, 700)

header = Label(root, bg="orange", width=300, height=2)
header.place(x=0, y=0)

h1 = Label(root, text="Titanic predictor", bg="orange", fg="black", font=('verdana', 13, 'bold'))  # подпись окна вверху
h1.place(x=135, y=5)

img = ImageTk.PhotoImage(Image.open('../art/titanic.png'))  # вставляем картинку logo
logo = Label(root, image=img, borderwidth=0)
logo.place(x=600, y=38)

# определяем начальное поле
Abbreviations = ['Mr.', 'Mrs.', 'Ms.', 'Miss.', 'Dr.', 'Major', 'Capt', 'Sir', 'Don', 'Master']
a = Label(root, text="Abbr.", font=('verdana', 10, 'bold'))  # надпись над полем ввода
a.place(x=50, y=120)
combobox_abr = TTK.Combobox(values=Abbreviations, height=9, width=7, state="readonly")
combobox_abr.place(x=50, y=145)

Name = TTK.Entry()
p = Label(root, text="Name", font=('verdana', 10, 'bold'))  # надпись над полем ввода
p.place(x=150, y=120)
Name.place(x=150, y=145)

SurName = TTK.Entry()
p = Label(root, text="SurName", font=('verdana', 10, 'bold'))  # надпись над полем ввода
p.place(x=350, y=120)
SurName.place(x=350, y=145)

# определяем первое поле
Ticket_class = ['1st = Upper', '2nd = Middle', '3rd = Lower']
q = Label(root, text="Ticket class", font=('verdana', 10, 'bold'))  # надпись над полем ввода
q.place(x=150, y=170)
combobox_tiket = TTK.Combobox(values=Ticket_class, state="readonly")
combobox_tiket.place(x=150, y=190)
# label = TTK.Label()
# label.place(x=250, y=170)
# combobox_tiket.bind("<<ComboboxSelected>>", selected)

# определяем второе поле
Sex = ['male', 'female']
w = Label(root, text="Sex", font=('verdana', 10, 'bold'))  # надпись над полем ввода
w.place(x=150, y=215)
combobox_sex = TTK.Combobox(values=Sex, state="readonly")
combobox_sex.place(x=150, y=235)


# определяем третье поле
Age_in_years = TTK.Entry()
e = Label(root, text="Age in years", font=('verdana', 10, 'bold'))  # надпись над полем ввода
e.place(x=150, y=260)
Age_in_years.place(x=150, y=285)

# определяем четвертое поле
# brother, sister, stepbrother, stepsister
# husband, wife
spouses_aboard_the_Titanic = ['0-alone', '1', '2', '3', '4', '5', '6', '7', '8']
r = Label(root, text="Number on board (brother, sister, stepbrother, stepsister, husband, wife)", font=('verdana', 10, 'bold'))  # надпись над полем ввода
r.place(x=50, y=310)
combobox_sibsp = TTK.Combobox(values=spouses_aboard_the_Titanic, state="readonly")
combobox_sibsp.place(x=150, y=335)

# определяем пятое поле
children_aboard_the_Titanic = ['0-alone', '1', '2', '3', '4', '5', '6']
t = Label(root, text="Number on board (mother, father, daughter, son, stepdaughter, stepson)", font=('verdana', 10, 'bold'))  # надпись над полем ввода
t.place(x=50, y=360)
combobox_parch = TTK.Combobox(values=children_aboard_the_Titanic, state="readonly")
combobox_parch.place(x=150, y=385)

# определяем шестое поле
Ticket_number = TTK.Entry()
y = Label(root, text="Ticket number", font=('verdana', 10, 'bold'))  # надпись над полем ввода
y.place(x=150, y=410)
Ticket_number.place(x=150, y=435)

# определяем седьмое поле
Passenger_fare = TTK.Entry()
u = Label(root, text="Passenger fare", font=('verdana', 10, 'bold'))  # надпись над полем ввода
u.place(x=150, y=460)
Passenger_fare.place(x=150, y=485)

# определяем восьмое поле
i = Label(root, text="Cabin letter", font=('verdana', 10, 'bold'))  # надпись над полем ввода
i.place(x=50, y=510)
Cabin_letter = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'T','without']
combobox_cabin_letter = TTK.Combobox(values=Cabin_letter, height=9, width=7, state="readonly")
combobox_cabin_letter.place(x=50, y=535)
Cabin_number = TTK.Entry()
i = Label(root, text="Cabin number 1-150", font=('verdana', 10, 'bold'))  # надпись над полем ввода
i.place(x=150, y=510)
Cabin_number.place(x=150, y=535)


# определяем девятое поле
Port_of_Embarkation = ['C = Cherbourg', 'Q = Queenstown', 'S = Southampton']
o = Label(root, text="Port of Embarkation", font=('verdana', 10, 'bold'))  # надпись над полем ввода
o.place(x=150, y=560)
combobox_embarked = TTK.Combobox(values=Port_of_Embarkation, state="readonly")
combobox_embarked.place(x=150, y=585)



# кнопка
predict = Button(root, text="Predict", padx=30, bg="orange", relief=RIDGE, borderwidth=1,
               font=('verdana', 10, 'bold'),
               cursor="hand2",
               command=get_text)
predict.place(x=150, y=620)

# Окно для вывода результата скачивания
Output = Text(root, height = 15, width = 37)
Output.place(x=470, y=400)
pl = PrintLogger(Output)
sys.stdout = pl


root.mainloop()

# survival	Survival	0 = No, 1 = Yes
# pclass	Ticket class	1 = 1st, 2 = 2nd, 3 = 3rd
# sex	Sex
# Age	Age in years
# sibsp	# of siblings / spouses aboard the Titanic
# parch	# of parents / children aboard the Titanic
# ticket	Ticket number
# fare	Passenger fare
# cabin	Cabin number
# embarked	Port of Embarkation
