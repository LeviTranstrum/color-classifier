from model import SimpleModel

def main():
    model = SimpleModel.load('large')

    model.run_training(10, visualize=True)

    model.test()

if __name__=="__main__":
    main()

