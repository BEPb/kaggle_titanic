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


# # обработка региона, указан перед названием сайта порнохаб (определение ссылка принадлежит или нет порнохабу)
# def ph_url_check():
#     url = url_dl.get()
#     parsed = urlparse.urlparse(url)
#     regions = ["www", "cn", "cz", "de", "es", "fr", "it", "nl", "jp", "pt", "pl", "rt"]
#     for region in regions:
#         if parsed.netloc == region + ".pornhub.com":
#             # print("PornHub url validated.")
#             Output.insert(END, 'PornHub url validated.')
#             return
#
#     # print("This is not a PornHub url.")
#     Output.insert(END, 'This is not a PornHub url.')
#     sys.exit()
#
# # проверка на существование данной страницы
# def ph_alive_check():
#     url = url_dl.get()
#     requested = requests.get(url)
#     if requested.status_code == 200:
#         print("and the URL is existing.")
#         # Output.insert(END, 'and the URL is existing')
#     else:
#         # print("but the URL does not exist.")
#         Output.insert(END, 'but the URL does not exist.')
#         sys.exit()
#     # return url
#
# # скачивание видео - срабатывает по нажатию кнопки скачать
# def Download_video():
#     url = url_dl.get()  # получаем ссылку на скачивание введенную в окно ввода
#     ph_url_check()  # проверка на принадлежность этой ссылки порнохабу
#     ph_alive_check()  # проверка на существование данной ссылки
#
#     outtmpl = 'Download/' + '%(title)s.%(ext)s'  # указываем папку для скачивания
#
#     # прописываем параметры youtube_dl
#     ydl_opts = {
#         'format': 'best',  # качество на скачивание установить лучшее
#         'outtmpl': outtmpl,
#         'nooverwrites': True,
#         'no_warnings': False,
#         'ignoreerrors': True,
#     }
#
#     # скачиваем файл
#
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

def selected(event):
    # получаем выделенный элемент
    selection = combobox.get()
    print(selection)
    label["text"] = f"вы выбрали: {selection}"


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
Name = TTK.Entry()
p = Label(root, text="Name", font=('verdana', 10, 'bold'))  # надпись над полем ввода
p.place(x=150, y=120)
Name.place(x=150, y=145)

# определяем первое поле
Ticket_class = ['1st = Upper', '2nd = Middle', '3rd = Lower']
q = Label(root, text="Ticket class", font=('verdana', 10, 'bold'))  # надпись над полем ввода
q.place(x=150, y=170)
combobox = TTK.Combobox(values=Ticket_class)
combobox.place(x=150, y=190)
label = TTK.Label()
label.place(x=250, y=170)
combobox.bind("<<ComboboxSelected>>", selected)

# определяем второе поле
Sex = ['male', 'female']
w = Label(root, text="Sex", font=('verdana', 10, 'bold'))  # надпись над полем ввода
w.place(x=150, y=215)
combobox = TTK.Combobox(values=Sex)
combobox.place(x=150, y=235)
label = TTK.Label()
label.place(x=250, y=170)
combobox.bind("<<ComboboxSelected>>", selected)

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
combobox = TTK.Combobox(values=spouses_aboard_the_Titanic)
combobox.place(x=150, y=335)
label = TTK.Label()
label.place(x=250, y=170)
combobox.bind("<<ComboboxSelected>>", selected)

# определяем пятое поле
children_aboard_the_Titanic = ['alone-0', '1', '2', '3', '4', '5', '6']
t = Label(root, text="Number on board (mother, father, daughter, son, stepdaughter, stepson)", font=('verdana', 10, 'bold'))  # надпись над полем ввода
t.place(x=150, y=360)
combobox = TTK.Combobox(values=children_aboard_the_Titanic)
combobox.place(x=150, y=385)
label = TTK.Label()
label.place(x=250, y=170)
combobox.bind("<<ComboboxSelected>>", selected)

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
combobox = TTK.Combobox(values=Port_of_Embarkation)
combobox.place(x=150, y=585)
label = TTK.Label()
label.place(x=250, y=170)
combobox.bind("<<ComboboxSelected>>", selected)


# кнопка
predict = Button(root, text="Predict", padx=30, bg="orange", relief=RIDGE, borderwidth=1,
               font=('verdana', 10, 'bold'),
               cursor="hand2")
predict.place(x=150, y=620)

# Окно для вывода результата скачивания
Output = Text(root, height = 22, width = 97)
Output.place(x=400, y=290)
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