import json
import matplotlib.pyplot
import numpy as np
import random
import matplotlib
import os

class Classifier:
    def __init__(self, name):
        self.name = name
        with open(f'./classifier/palettes/{name}.json') as palette:
            self.palette = json.load(palette)

    def _color_error(self, color1, color2):
        return np.sqrt((color1[0] - color2[0])**2 + (color1[1] - color2[1])**2 + (color1[2] - color2[2])**2)

    def classify(self, color):
        lowest_score = 1000.0
        label = None
        for index, value in enumerate(self.palette):
            score = self._color_error(color, value[0])
            if score < lowest_score:
                lowest_score = score
                label = int(index)
        return label

    def random_color(self, seed=None):
        random.seed(seed)
        return [random.randint(0,255), random.randint(0,255), random.randint(0,255)]

    def visualize(self, colors):
        # Determine grid size for plotting (simple square layout)
        n = len(colors)
        grid_size = int(np.ceil(np.sqrt(n)))

        fig, axs = matplotlib.pyplot.subplots(grid_size, grid_size, figsize=(grid_size*1.1, grid_size*1.1))
        axs = axs.flatten()  # Flatten the array of axes for easier indexing

        # Plot each color
        for idx, color in enumerate(colors):
            data = np.zeros((10, 10, 3), dtype=np.uint8)
            data[:, :, :] = color
            axs[idx].imshow(data, interpolation='nearest')
            axs[idx].axis('off')  # Hide the axes
            axs[idx].set_title(self.palette[self.classify_hard(color)][0])

        # Hide any unused subplots if n is not a perfect square
        for idx in range(n, grid_size * grid_size):
            axs[idx].axis('off')

        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.show()

    def show_palette(self):
        self.visualize([item[1] for item in enumerate(self.palette)])

    def generate_data(self, num):
        traindata = []
        for i in range(num):
            color = self.random_color()
            traindata.append ({'color': color, 'label': self.classify_hard(color)})
            
        testdata = []
        for key, value in self.palette.items():
            testdata.append({'color': value['color'], 'label': key})
        
        os.makedirs(os.path.dirname(f"./data/{self.name}/train.json"), exist_ok=True)
        os.makedirs(os.path.dirname(f"./data/{self.name}/test.json"), exist_ok=True)

        with open(f"./data/{self.name}/train.json", mode="w") as trainfile:
            json.dump(traindata, trainfile)
        
        with open(f"./data/{self.name}/test.json", mode="w") as testfile:
            json.dump(testdata, testfile)
        
def test():
    classifier = Classifier('geometric27-narrow')
    classifier.show_palette()
    colors = [classifier.random_color() for _ in range(100)]
    classifier.visualize(colors)

    classifier.generate_data(10000)

test()