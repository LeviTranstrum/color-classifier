import torch
from models import SimpleModel

def main():
    # Initialize model
    model = SimpleModel('geometric27-narrow')

    model.run_training(3, visualize=True)

    model.run_validation('Trained Model Outputs')

if __name__=="__main__":
    main()

