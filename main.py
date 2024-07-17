import torch
from model import SimpleModel

def main():
    # Initialize model
    model = SimpleModel('geometric27-narrow')

    model.run_training(1, visualize=False)

    # model.run_validation('Trained Model Outputs')

    model.save()

if __name__=="__main__":
    main()

