################################################### IMPORTS ###################################################################

import os
import cv2
import numpy as np
from Image import return_type
from Image import SignalType
from tqdm import tqdm
from Algoritmos import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

#############################################################################################################################
############################################# VARIABLES_GLOBALES ##############################################################
default_mask_dimension = (32, 32)
#############################################################################################################################


class Warehouse:

    def __init__(self):
        self.clasificadores_binarios = {SignalType.PROHIBIDO: [], SignalType.PELIGRO: [], SignalType.STOP: [
        ], SignalType.DIRECCION_PROHIBIDA: [], SignalType.CEDA_EL_PASO: [], SignalType.DIRECCION_OBLIGATORIA: []}
        self.train_images_info = {}
        self.train_images = {SignalType.PROHIBIDO: [], SignalType.PELIGRO: [], SignalType.STOP: [
        ], SignalType.DIRECCION_PROHIBIDA: [], SignalType.CEDA_EL_PASO: [], SignalType.DIRECCION_OBLIGATORIA: [], SignalType.NO_SEÑAL: []}
        self.test_images = {}
        self.validation_set = {}
        self.detector = Detector()
        self.knn = KNeighborsClassifier()
        self.pca = PCA(n_components=9)


############################################CARGA DE DATOS DE ENTRENAMIENTO###################################################

####################### METODO PRINCIPAL #######################################################################################


    def load_train_images(self, path):
        '''
        Lee el archvio gt.txt, extrae de cada imagen las regiones de interes,clasifica cada señal de cada region y  las almacena en el diccionario train_images_info
        Parameters
        ----------
        path : string
            ruta en la que se encuentra el archivo gt.txt

        Raises
        ------
        Exception
            En caso de que el directorio proporcionado no exista
        '''
        if not os.path.isdir(path):
            raise Exception(
                f'File {path + "/gt.txt"} No se ha encontrado el fichero gt.txt!')
        print('Cargando datos de entrenamiento...')
        gt_file = path + '/gt.txt'
        num_lines = 0
        with open(gt_file) as f:
            for _ in f:
                num_lines += 1
        with open(gt_file, 'r') as f:
            for i, line in enumerate(tqdm(f, total=num_lines)):
                image_info = self.line_to_img(line, path)
                if image_info[0] in self.train_images_info:
                    self.train_images_info[image_info[0]].append(image_info[1])
                else:
                    self.train_images_info[image_info[0]] = [image_info[1]]

        self.generate_incorrect_data(path)
####################### METODOS AUXILIARES #####################################################################################

    def line_to_img(self, line: str, path):
        '''
        Extrae la informacion de cada linea y con ella crea el objeto imagen ya clasificado

        Parameters
        ----------
        line : str
            Cadena con la informacion detallada de la imagen y la region de interes
        path : str
            Ubicacion donde se encuentra almacenada la imagen

        Returns
        -------
        Image
            Devuelve un objeto imagen que encapsula tanto la imagen binaria como su tipo
        '''
        args = line.split(';')
        master_image = cv2.imread(path + '/' + args[0])
        minx = int(args[1])
        miny = int(args[2])
        maxx = int(args[3])
        maxy = int(args[4])
        type_value = int(args[5])
        signal_type = return_type(type_value)
        name = args[0]
        rectangle = (minx, miny, maxx - minx, maxy - miny)
        if signal_type in self.train_images:
            crop_image = master_image[miny:maxy, minx:maxx]
            self.train_images[signal_type].append(
                crop_image)
        return name, rectangle

    def generate_incorrect_data(self, path):
        '''
        Genera imagenes erroneas para el entrenamiento de los clasificadores binarios

        Parameters
        ----------
        path : string
            ruta en la que se encuentran las imagenes de entrenamiento
        '''

        incorrect_data = []
        print('Generando imagenes de entrenamiento erroneas...')
        for filename in tqdm(os.listdir(path)):
            if filename.endswith(".jpg"):
                img = cv2.imread(path+'/' + filename)
                #trash_regions = self.detector.complete_detect_regions(img)
                trash_regions = self.detector.ligth_detect_regions(img)
                self.filter_trash_regions(
                    incorrect_data, filename, img, trash_regions)
        self.train_images[SignalType.NO_SEÑAL] = incorrect_data

    def filter_trash_regions(self, incorrect_data, filename, img, trash_regions):
        '''
        Filtra las regiones de interes que no sean de señal
        Parameters
        ----------
        incorrect_data : list
            _description_
        filename : string
            _description_
        img : Image
            _description_
        trash_regions : list
            _description_
        '''
        if filename in self.train_images_info:
            signal_regions = self.train_images_info[filename]
            for trash_region in trash_regions:
                condition = True
                for signal_region in signal_regions:
                    if(self.detector.get_IoU(trash_region, signal_region) > 0.1):
                        condition = False
                if condition:
                    incorrect_data.append(
                        img[trash_region[1]:trash_region[1]+trash_region[3], trash_region[0]:trash_region[0]+trash_region[2]])
        else:
            for trash_region in trash_regions:
                incorrect_data.append(
                    img[trash_region[1]:trash_region[1]+trash_region[3], trash_region[0]:trash_region[0]+trash_region[2]])
###############################################################################################################################

#################################################### CARGA DE DATOS DE TEST ######################################################
    def load_test_images(self, path):
        '''
        Lee todas las imagenes del directorio test y las almacena en el diccionario test_images
        '''
        print('Cargando imagenes de prueba...')
        for filename in tqdm(os.listdir(path)):
            if filename.endswith(".jpg"):
                img = cv2.imread(path+'/' + filename)
                # delete the extension of the image
                name = filename.split('.')[0]
                self.test_images[name] = img
###############################################################################################################################

#################################################### GUARDADO DE DATOS  ######################################################

####################### METODO PRINCIPAL #######################################################################################

    def save_images(self, processed_images: dict):
        '''
        Guarda las imagenes procesadas en el directorio resultado_imgs
        '''
        self.save_processed_images(processed_images)
        self.save_images_info(processed_images)
####################### METODOS AUXILIARES #####################################################################################

    def save_images_info(self, processed_images):
        print('Guardando informacion de las imagenes procesadas...')
        if not os.path.exists('resultado.txt'):
            open('resultado.txt', 'w').close()
        with open('resultado.txt', 'w') as f:
            for name, img_data in tqdm(processed_images.items()):
                for region in img_data[1]:
                    square = region[0]
                    info = region[1]
                    x, y, w, h = square
                    x2 = x + w
                    y2 = y + h
                    f.write(
                        f'{name}.jpg;{x};{y};{x2};{y2};{info[0].value};{info[1]}\n')
                    # f'{name}.jpg;{region[0][0]};{region[0][1]};{region[0][2]};{region[0][3]};{region[1][0].value};{region[1][1]}\n')

    def save_processed_images(self, processed_images):
        print('Guardando imagenes procesadas...')
        if not os.path.exists('./resultado_imgs'):
            os.makedirs('./resultado_imgs')
        for name, img_data in tqdm(processed_images.items()):
            cv2.imwrite(f'./resultado_imgs/{name}.jpg', img_data[0])
###############################################################################################################################

#################################################### TRATAMIENTO DE DATOS LDA BAYESIANO #########################################

####################### METODO PRINCIPAL #######################################################################################
    def data_treatment(self, classifier_type):
        '''

        Realiza el tratamiento de los datos para el entrenamiento de los clasificadores binarios y su entrenamiento

        Parameters
        ----------
        classifier_type : string
            tipo de clasificador a utilizar
        '''
        self.select_preprocessing_data_treatmet(classifier_type)
        self.save_validation_set()
        fw, sw, tw = classifier_type
        if tw == 'BAYES':
            self.apply_lda_bayes()
        elif tw == 'KNN':
            self.apply_pca_knn()

####################### METODOS AUXILIARES #####################################################################################

    def select_preprocessing_data_treatmet(self, clasification_type):
        if clasification_type[0] == 'HOG':
            self.apply_hog()
        elif clasification_type[0] == 'GRAY':
            self.apply_gray_vectorization()
        elif clasification_type[0] == 'RGB':
            self.apply_rgb_vectorization()
        elif clasification_type[0] == 'CANY':
            self.apply_cany()

    def apply_cany(self):
        '''
        Aplica la transformacion de canny a todas las imagenes de entrenamiento
        '''
        print('Convirtiendo las imagenes de entrenamiento vectores de caracteristicas de bordes ...')
        for key in tqdm(self.train_images.keys()):
            imagenes_señal = self.train_images[key]
            features_vectors = []
            for img in imagenes_señal:
                img = self.detector.resize_img(img)
                cany_vector = np.array(cv2.Canny(img, 100, 200)).flatten()
                features_vectors = features_vectors + [cany_vector]
            self.train_images[key] = features_vectors

    def apply_rgb_vectorization(self):
        '''
        Aplica la transformacion de vectorizacion de caracteristicas de color a todas las imagenes de entrenamiento
        '''
        print('Convirtiendo las imagenes de entrenamiento en vectores de rgb...')
        for key in tqdm(self.train_images.keys()):
            imagenes_señal = self.train_images[key]
            features_vectors = []
            for img in imagenes_señal:
                img = self.detector.resize_img(img)
                rbg_vector = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                rbg_vector = rbg_vector.flatten()
                features_vectors = features_vectors + [rbg_vector]
            self.train_images[key] = features_vectors

    def apply_gray_vectorization(self):
        '''
        Aplica la transformacion de vectorizacion de caracteristicas de color a todas las imagenes de entrenamiento

        '''
        print('Convirtiendo las imagenes de entrenamiento en vectores de gris...')
        for key in tqdm(self.train_images.keys()):
            imagenes_señal = self.train_images[key]
            features_vectors = []
            for img in imagenes_señal:
                img = self.detector.resize_img(img)
                gray_vector = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                gray_vector = gray_vector.flatten()
                features_vectors = features_vectors + [gray_vector]
            self.train_images[key] = features_vectors

    def apply_lda_bayes(self):
        '''
        Aplica la transformacion de LDA las imagenes y genera el clasificador bayesiano para cada una de las señales
        '''
        print('Aplicando reduccion de dimensionalidad a las imagenes de entrenamiento con el algoritmo LDA y generando Clasificadores binarios ...')
        no_signal_data = self.train_images[SignalType.NO_SEÑAL]
        del self.train_images[SignalType.NO_SEÑAL]
        for key in tqdm(self.train_images.keys()):
            lda = LinearDiscriminantAnalysis()
            signal_data = self.train_images[key]
            signal_data = signal_data+no_signal_data
            signal_data = np.array(signal_data, dtype=np.float32)
            aux = np.zeros(len(no_signal_data), dtype=np.int16)
            labels = np.ones(
                len(self.train_images[key]), dtype=np.int16)
            labels = labels*key.value
            labels = np.append(labels, aux)
            labels = labels.astype(np.float32)
            self.clasificadores_binarios[key] = lda.fit(signal_data, labels)

    def apply_pca_knn(self):
        '''
        Aplica la transformacion de LDA las imagenes y genera el clasificador bayesiano para cada una de las señales
        '''
        print('Aplicando reduccion de dimensionalidad a las imagenes de entrenamiento con el algoritmo PCA y generando Clasificadores binarios ...')
        total_data = []
        total_labels = np.array([])
        for key in tqdm(self.train_images.keys()):
            signal_data = self.train_images[key]
            for element in signal_data:
                total_data.append(element)
            labels = np.ones(len(signal_data))
            labels = labels*[key.value]
            total_labels = np.append(total_labels, labels)
        total_data_reduced = self.pca.fit_transform(total_data, total_labels)
        self.knn.fit(total_data_reduced, total_labels)

    def save_validation_set(self):
        '''
        Guarda el conjunto de validacion en un diccionario para la posterior evaluacion
        '''
        print('Almacenando subconjunto de datos de validacion...')
        for key in tqdm(self.train_images.keys()):
            self.validation_set[key] = self.train_images[key][int(
                len(self.train_images[key]) * 0.1):]

    def apply_hog(self):
        '''
        Aplica la transformacion de HOG a todas las imagenes de entrenamiento
        '''
        print('Aplicando el algoritmo HOG a las imagenes de entrenamiento...')
        for key in tqdm(self.train_images.keys()):
            imagenes_señal = self.train_images[key]
            features_vectors = []
            for img in imagenes_señal:
                hog_result = self.detector.hog(img)
                features_vectors = features_vectors + [hog_result]
            self.train_images[key] = features_vectors
###############################################################################################################################
