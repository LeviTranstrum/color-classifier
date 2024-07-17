import torch
from model import SimpleModel

def main():
    model = SimpleModel.load('geometric27-narrow')

    model.run_training(3, visualize=True)

    model.run_validation('Trained Model Outputs')

    model.save()

if __name__=="__main__":
    main()

