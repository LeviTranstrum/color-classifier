import json
import matplotlib.pyplot
import numpy as np
import random
import matplotlib
import os

class Generator:
    def __init__(self, name):
        self.name = name
        with open(f'./generator/palettes/{name}.json') as palette:
            self.palette = json.load(palette)

    def _color_error(self, color1, color2):
        return np.sqrt((color1[0] - color2[0])**2 + (color1[1] - color2[1])**2 + (color1[2] - color2[2])**2)

    def classify(self, color):
        lowest_score = 1000.0
        label = None
        for index, value in enumerate(self.palette):
            score = self._color_error(color, value[1])
            if score < lowest_score:
                lowest_score = score
                label = int(index)
        return label

    def random_color(self, seed=None):
        random.seed(seed)
        return [random.randint(0,255), random.randint(0,255), random.randint(0,255)]

    def visualize(self, colors, labels = None, title=None):
        if labels is None:
            labels=[]
            for color in enumerate(colors):
                labels.append([self.index_to_label(self.classify(color))])

        if len(labels) != len(colors):
            raise ValueError("colors and labels must have the same number of elements")

        # Determine grid size for plotting (simple square layout)
        n = len(colors)
        grid_size = int(np.ceil(np.sqrt(n)))

        fig, axs = matplotlib.pyplot.subplots(grid_size, grid_size, figsize=(grid_size*1.1, grid_size*1.1))
        axs = axs.flatten()  # Flatten the array of axes for easier indexing
        if title is not None:
            fig.canvas.manager.set_window_title(title)

        # Plot each color
        for i in range(len(colors)):
            data = np.zeros((10, 10, 3), dtype=np.uint8)
            data[:, :, :] = colors[i]
            axs[i].imshow(data, interpolation='nearest')
            axs[i].axis('off')  # Hide the axes
            axs[i].set_title(labels[i])

        # Hide any unused subplots if n is not a perfect square
        for idx in range(n, grid_size * grid_size):
            axs[idx].axis('off')

        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.show()

    def show_palette(self):
        self.visualize([self.palette[i][1] for i in range(len(self.palette))],
                       [self.palette[i][0] for i in range(len(self.palette))])

    def index_to_label(self, index):
        return self.palette[index][0]

    def generate_data(self, num=100000):
        train_data = []
        for i in range(num):
            color = self.random_color()
            train_data.append ([color, self.classify(color)])

        validate_data = []
        for i in range(len(self.palette)):
            validate_data.append([self.palette[i][1], i])

        train_path = f'./data/{self.name}/train.json'
        validate_path = f'./data/{self.name}/validate.json'

        os.makedirs(os.path.dirname(train_path), exist_ok=True)
        os.makedirs(os.path.dirname(validate_path), exist_ok=True)

        with open(train_path, mode="w") as train_file:
            json.dump(train_data, train_file)

        with open(validate_path, mode="w") as validate_file:
            json.dump(validate_data, validate_file)

    @classmethod
    def test(cls):
        print('testing generator')
        generator = Generator('geometric27-narrow')
        generator.show_palette()
        colors = [generator.random_color() for _ in range(100)]
        generator.visualize(colors)
