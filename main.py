from model import SimpleModel, MediumModel, LargeModel
import torch

def main():
    model = LargeModel.load('large')

    model.run_training(10, visualize=False)

    model.run_validation()

    model.info()

if __name__=="__main__":
    main()

