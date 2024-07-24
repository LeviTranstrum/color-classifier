from model import SimpleModel, MediumModel, LargeModel

def main():
    model = LargeModel.load('large')

    # model.run_training(10, visualize=False)

    # model.test()

    model.run_validation()
if __name__=="__main__":
    main()

