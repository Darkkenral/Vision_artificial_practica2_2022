import argparse
from sys import argv, stdout
from Almacen import *
from Algoritmos import *

if __name__ == "__main__":

    wh = Warehouse()
    d = Detector()

    parser = argparse.ArgumentParser(
        description='Entrena sober train y ejecuta el clasificador sobre imgs de test')
    parser.add_argument(
        '--train_path', type=str, default="./train", help='Path al directorio de imgs de train')
    parser.add_argument(
        '--test_path', type=str, default="./test", help='Path al directorio de imgs de test')
    parser.add_argument(
        '--classifier', type=str, default="HOG_LDA_BAYES", help='String con el nombre del clasificador')

    args = parser.parse_args()
    argv = args.train_path, args.test_path, args.classifier

    # check if the number of arguments is correct print the help if not
    if len(argv) != 3:
        parser.print_help()
        exit(1)

    if args.classifier not in ["HOG_LDA_BAYES", "GRAY_LDA_BAYES"]:
        print("Error: el clasificador no es valido")
        parser.print_help()
        exit(1)
    if not os.path.isdir(args.train_path):
        print("Error: el directorio de train no existe")
        parser.print_help()
        exit(1)
    if not os.path.isdir(args.test_path):
        print("Error: el directorio de test no existe")
        parser.print_help()
        exit(1)

    # Cargar los datos de entrenamiento
    wh.load_train_images(args.train_path)
    # Tratamos los datos en funcion del clasificador
    wh.data_treatment_LDA_BAYES(args.classifier.upper())
    # Cargar los datos de test
    wh.load_test_images(args.test_path)
    # Clasificar los datos de test y almacenamiento
    wh.save_images(d.multi_class_classifier_LDA_BAYES(
        wh.test_images, wh.clasificadores_binarios, args.classifier.upper()))
    # Evaluar el clasificador
    d.evaluate_classifier(wh.validation_set, wh.clasificadores_binarios)


# python3 main.py --train_path="train_jpg" --test_path="test_alumnos_jpg" --classifier="HOG_LDA_BAYES"
