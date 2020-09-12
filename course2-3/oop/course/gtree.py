import tkinter as tk


class GraphicalTree:
    def __init__(self, tree, title=None, width=800, height=600):
        self.__tree = tree
        self.__window = tk.Tk()
        self.__window.title(title)
        self.__canvas = tk.Canvas(self.__window, width=width, height=height)
        self.__canvas.configure(background="white")
        self.__canvas.pack()
        self.__canvas.bind("<Button-4>", self.__scroll)
        self.__canvas.bind("<Button-5>", self.__scroll)
        self.__canvas.bind("<ButtonPress-1>", self.__drag_start)
        self.__canvas.bind("<B1-Motion>", self.__drag_move)
        self.__scale = 1

    def __scroll(self, event):
        if event.num == 4 or event.delta == 120:
            self.__canvas.scale("all", self.__canvas.canvasx(event.x), self.__canvas.canvasy(event.y), 1.1, 1.1)
            self.__scale *= 1.1
        elif event.num == 5 or event.delta == -120:
            self.__canvas.scale("all", self.__canvas.canvasx(event.x), self.__canvas.canvasy(event.y), 0.9, 0.9)
            self.__scale *= 0.9
        self.__canvas.itemconfigure("text_data", font=(None, round(self.__scale * 20)))

    def __drag_start(self, event):
        self.__canvas.scan_mark(event.x, event.y)

    def __drag_move(self, event):
        self.__canvas.scan_dragto(event.x, event.y, gain=1)

    def __draw_vertex(self, data, x, y):
        radius = 20
        self.__canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="white")
        self.__canvas.create_text(x, y, text=data, font=(None, round(self.__scale * 20)), tag="text_data")

    def __draw_tree(self, vertex, level=1):
        cell_size = 40
        if vertex:
            if vertex.left:
                self.__canvas.create_line((vertex.data + 1) * cell_size, level * cell_size,
                                          (vertex.left.data + 1) * cell_size, (level + 1) * cell_size)
            if vertex.right:
                self.__canvas.create_line((vertex.data + 1) * cell_size, level * cell_size,
                                          (vertex.right.data + 1) * cell_size, (level + 1) * cell_size)

            self.__draw_vertex(vertex.data, (vertex.data + 1) * cell_size, level * cell_size)
            self.__draw_tree(vertex.left, level + 1)
            self.__draw_tree(vertex.right, level + 1)

    def start(self):
        self.update()
        self.__window.mainloop()

    def update(self):
        self.__canvas.delete("all")
        self.__draw_tree(self.__tree.root)
        self.__canvas.update()
