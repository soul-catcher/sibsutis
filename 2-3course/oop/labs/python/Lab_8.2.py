from tkinter import *
import tkinter.messagebox

root = Tk()
root.geometry("200x200+500+300")

mainMenu = Menu()

fileMenu = Menu(mainMenu, tearoff=0)
fileMenu.add_command(label="Save F2")
fileMenu.add_command(label="Exit Alx+X", command=lambda: root.quit())

helpMenu = Menu(mainMenu, tearoff=0)
helpMenu.add_command(label="About", command=lambda: tkinter.messagebox.showinfo("Information", "Приложение\n"
                                                                                               "Версия 1.0"))
helpMenu.add_command(label="Help F1")

mainMenu.add_cascade(label='File', menu=fileMenu)
mainMenu.add_cascade(label='Help', menu=helpMenu)
root.configure(menu=mainMenu)

label = Label(text="Ожидается ввод текста")
label.pack()

entry = Entry()
entry.pack()
entry.bind("<Key>", lambda key: label.configure(text="Идёт ввод текста"))
entry.focus()

button1 = Button(text="Сброс", command=lambda: (entry.delete(0, 'end'), label.configure(text="Ожидается ввод текста")))
button2 = Button(text="Выход", command=lambda: root.quit())
button1.pack()
button2.pack()

new_order = (label, button2, button1)
for widget in new_order:
    widget.lift()

root.mainloop()
