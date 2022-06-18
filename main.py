import argparse
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

    # Cargar los datos de entrenamiento
    wh.load_train_images(args.train_path)
    # Seleccionar el clasificador
    if args.classifier == "HOG_LDA_BAYES":
        # Tratamiento de los datos y entrenamiento  del clasificador
        wh.data_treatment_HOG_LDA_BAYES()
        # Cargar los datos de test
        wh.load_test_images(args.test_path)
        # Clasificar los datos de test y almacenamiento
        wh.save_images(d.multi_class_classifier_HOG_LDA_BAYES(
            wh.test_images, wh.clasificadores_binarios))
        # Evaluar el clasificador
        d.evaluate_classifier(wh.validation_set, wh.clasificadores_binarios)
    else:
        raise ValueError('Tipo de clasificador incorrecto')

# python3 main.py --train_path="/home/miguel/Documentos/Workspace/Vision_artificial_practica2_2022/train_jpg" --test_path="/home/miguel/Documentos/Workspace/Vision_artificial_practica2_2022/test_alumnos_jpg" --classifier=HOG_LDA_BAYES
