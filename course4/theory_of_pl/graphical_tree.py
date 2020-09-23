import tkinter as tk


class Vertex:
    def __init__(self, data, *children):
        self.data = data
        self.children = list(children)


class GraphicalTree:
    def __init__(self, tree, title=None, width=800, height=600):
        self._tree = tree
        self._window = tk.Tk()
        self._window.title(title)
        self._canvas = tk.Canvas(self._window, width=width, height=height)
        self._canvas.pack()
        self._canvas.bind("<Button-4>", self._scroll)
        self._canvas.bind("<Button-5>", self._scroll)
        self._canvas.bind("<ButtonPress-1>", self._drag_start)
        self._canvas.bind("<B1-Motion>", self._drag_move)

    def _scroll(self, event):
        if event.num == 4 or event.delta == 120:
            self._canvas.scale("all", event.x, event.y, 1.1, 1.1)
        elif event.num == 5 or event.delta == -120:
            self._canvas.scale("all", event.x, event.y, 0.9, 0.9)

    def _drag_start(self, event):
        self._canvas.scan_mark(event.x, event.y)

    def _drag_move(self, event):
        self._canvas.scan_dragto(event.x, event.y, gain=1)
        self._canvas.update()

    def _draw_vertex(self, data, x, y):
        radius = 15
        self._canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="white")
        self._canvas.create_text(x, y, text=data, font=(None, round(20)))

    def _draw_tree(self, vertex, level=1, margin=1):
        width = 0
        cell_size = 40
        for child in vertex.children:
            self._canvas.create_line(margin * cell_size, level * cell_size, (margin + width) * cell_size, (level + 1) * cell_size)
            width += self._draw_tree(child, level + 1, margin + width)
        self._draw_vertex(vertex.data, margin * cell_size, level * cell_size)

        return width if width else 1

    def start(self):
        self.update()
        self._window.mainloop()

    def update(self):
        self._canvas.delete("all")
        self._draw_tree(self._tree)
        self._canvas.update()
