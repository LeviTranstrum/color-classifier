import json
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import namedtuple

class Classifier:
    def __init__(self, palette, name):
        self.palette = palette
        self.name = name

    def _color_error(self, color1, color2):
        return np.sqrt((color1[0] - color2[0])**2 + (color1[1] - color2[1])**2 + (color1[2] - color2[2])**2)

    def classify_hard(self, color):
        lowest_score = 1000.0
        label = -1
        for key, value in self.palette.items():
            score = self._color_error(color, value["color"])
            if score < lowest_score:
                lowest_score = score
                label = key
        return label

    def classify_soft(self, color):
        scores = {}
        for key, value in self.palette.items():
            scores[key] = 1 - self._color_error(color, value.color) / 442.0
        print(scores)
        return scores

    def classify_n(self, color, n):
        if n > len(self.palette):
            raise ValueError("n cannot be greater than the number of colors in palette")
        label = ""
        scores = self.classify_soft(color)
        for _ in range(n):
            max_score_label = max(scores, key=scores.get)
            label += max_score_label + "/"
            del scores[max_score_label]
        return label[:-1]

    def random_color(self, seed=None):
        random.seed(seed)
        return [random.randint(0,255), random.randint(0,255), random.randint(0,255)]

    def visualize(self, colors):
        # Determine grid size for plotting (simple square layout)
        n = len(colors)
        grid_size = int(np.ceil(np.sqrt(n)))

        fig, axs = plt.subplots(grid_size, grid_size, figsize=(grid_size*1.1, grid_size*1.1))
        axs = axs.flatten()  # Flatten the array of axes for easier indexing

        # Plot each color
        for idx, color in enumerate(colors):
            data = np.zeros((10, 10, 3), dtype=np.uint8)
            data[:, :, :] = color
            axs[idx].imshow(data, interpolation='nearest')
            axs[idx].axis('off')  # Hide the axes
            axs[idx].set_title(self.classify_hard(color))

        # Hide any unused subplots if n is not a perfect square
        for idx in range(n, grid_size * grid_size):
            axs[idx].axis('off')

        plt.tight_layout()
        plt.show()

    def show_palette(self):
        self.visualize([item["color"] for item in self.palette.values()])

    def generate_data(self, num):
        data = []
        for i in range(num):
            color = self.random_color()
            data.append ({'color': color, 'label': self.classify_hard(color)})
        with open(f"./data/{self.name}.json", mode="w") as datafile:
            json.dump(data, datafile)

with open('./classifier/palettes/geometric27-narrow.json') as palette:
    classifier = Classifier(json.load(palette), 'geometric27-narrow')
    classifier.show_palette()
    colors = [classifier.random_color() for _ in range(100)]
    classifier.visualize(colors)

    classifier.generate_data(10000)