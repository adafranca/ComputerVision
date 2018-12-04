from prepare_data import getdata
import model

def main():
        training_gen = getdata()
        model.train_generator(training_gen)

main()