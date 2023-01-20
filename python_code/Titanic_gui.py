"""
Python 3.9 prediction program will you survive on the titanic or not
File name Titanic_gui.py

Version: 0.1
Author: Andrej Marinchenko
Date: 2022-04-19
"""

#  установка необходимых библиотек
import requests
import sys
import urllib.parse as urlparse

from tkinter import *
import tkinter as tk
import tkinter.ttk as TTK
from PIL import ImageTk, Image



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
    cabin_t = Cabin_number.get()
    embarked_t = combobox_embarked.get()
    # label['text'] = name_t
    print(abr_t, ' ', name_t, ' ', surname_t)
    print(pclass_t)
    print(sex_t)
    print(age_t)
    print(sibsp_t)
    print(parch_t)
    print(tiket_num_t)
    print(fare_t)
    print(cabin_t)
    print(embarked_t)


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
Abbreviations = ['Mr.', 'Mrs.', 'Ms.', 'Miss.', 'Dr.', 'Major', 'Capt', 'Sir', 'Don']
a = Label(root, text="Abbr.", font=('verdana', 10, 'bold'))  # надпись над полем ввода
a.place(x=40, y=120)
combobox_abr = TTK.Combobox(values=Abbreviations)
combobox_abr.place(x=40, y=145)

Name = TTK.Entry()
p = Label(root, text="Name", font=('verdana', 10, 'bold'))  # надпись над полем ввода
p.place(x=150, y=120)
Name.place(x=150, y=145)

SurName = TTK.Entry()
p = Label(root, text="SurName", font=('verdana', 10, 'bold'))  # надпись над полем ввода
p.place(x=150, y=120)
SurName.place(x=350, y=145)

# определяем первое поле
Ticket_class = ['1st = Upper', '2nd = Middle', '3rd = Lower']
q = Label(root, text="Ticket class", font=('verdana', 10, 'bold'))  # надпись над полем ввода
q.place(x=150, y=170)
combobox_tiket = TTK.Combobox(values=Ticket_class)
combobox_tiket.place(x=150, y=190)
# label = TTK.Label()
# label.place(x=250, y=170)
# combobox_tiket.bind("<<ComboboxSelected>>", selected)

# определяем второе поле
Sex = ['male', 'female']
w = Label(root, text="Sex", font=('verdana', 10, 'bold'))  # надпись над полем ввода
w.place(x=150, y=215)
combobox_sex = TTK.Combobox(values=Sex)
combobox_sex.place(x=150, y=235)


# определяем третье поле
Age_in_years = TTK.Entry()
e = Label(root, text="Age in years", font=('verdana', 10, 'bold'))  # надпись над полем ввода
e.place(x=150, y=260)
Age_in_years.place(x=150, y=285)

# определяем четвертое поле
# brother, sister, stepbrother, stepsister
# husband, wife
spouses_aboard_the_Titanic = ['alone-0', '1', '2', '3', '4', '5', '6', '7', '8']
r = Label(root, text="Number on board (brother, sister, stepbrother, stepsister, husband, wife)", font=('verdana', 10, 'bold'))  # надпись над полем ввода
r.place(x=150, y=310)
combobox_sibsp = TTK.Combobox(values=spouses_aboard_the_Titanic)
combobox_sibsp.place(x=150, y=335)

# определяем пятое поле
children_aboard_the_Titanic = ['alone-0', '1', '2', '3', '4', '5', '6']
t = Label(root, text="Number on board (mother, father, daughter, son, stepdaughter, stepson)", font=('verdana', 10, 'bold'))  # надпись над полем ввода
t.place(x=150, y=360)
combobox_parch = TTK.Combobox(values=children_aboard_the_Titanic)
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
Cabin_number = TTK.Entry()
i = Label(root, text="Cabin number", font=('verdana', 10, 'bold'))  # надпись над полем ввода
i.place(x=150, y=510)
Cabin_number.place(x=150, y=535)

# определяем девятое поле
Port_of_Embarkation = ['C = Cherbourg', 'Q = Queenstown', 'S = Southampton']
o = Label(root, text="Port of Embarkation", font=('verdana', 10, 'bold'))  # надпись над полем ввода
o.place(x=150, y=560)
combobox_embarked = TTK.Combobox(values=Port_of_Embarkation)
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