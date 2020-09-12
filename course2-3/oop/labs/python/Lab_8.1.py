from tkinter import *

root = Tk()
root.geometry("200x200+500+300")
label = Label(text=1)
label.pack()

entryText = StringVar()
entryText.set("Привет, студент")

entry = Entry(textvariable=entryText)
entry.pack()

button = Button(command=lambda: label.configure(text=label.cget("text") + 1))
button.pack()

root.mainloop()
