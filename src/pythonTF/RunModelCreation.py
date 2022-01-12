"""
Run the model creation
"""
from CreateData import ResistorDataSet
from CreateModel import ANN


def main():
    dataset = ResistorDataSet()

    dataset.create_data()

    model = ANN(save_path='/home/connor/Documents/DeepSim/CUDA/TFCPP/models',
                name='resistor')

    model.create_model()

    model.train_model(dataset=dataset)

    model.save_model()


if __name__ == '__main__':
    main()