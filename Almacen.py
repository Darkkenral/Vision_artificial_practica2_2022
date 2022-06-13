
from multiprocessing import Condition
import os
import cv2
from cv2 import rectangle
import numpy as np
from Image import return_type
from Image import SignalType
from tqdm import tqdm
from Algoritmos import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
default_mask_dimension = (32, 32)


class Warehouse:

    def __init__(self):
        self.clasificadores_binarios = {SignalType.PROHIBIDO: [], SignalType.PELIGRO: [], SignalType.STOP: [
        ], SignalType.DIRECCION_PROHIBIDA: [], SignalType.CEDA_EL_PASO: [], SignalType.DIRECCION_OBLIGATORIA: [], SignalType.NO_SEÑAL: []}
        self.train_images_info = {}
        self.test_images = {}
        self.detector = Detector()

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
        type = int(args[5])
        signal_type = return_type(type)
        name = args[0]
        # return the rectangle with the format (x, y, width, height)
        rectangle = (minx, miny, maxx - minx, maxy - miny)
        if signal_type in self.clasificadores_binarios.keys():
            crop_image = master_image[miny:maxy, minx:maxx]
            self.clasificadores_binarios[signal_type].append(
                crop_image)
        return name, rectangle

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
            raise Exception(f'File {path + "/gt.txt"} not found!')
        print('Cargando datos de entrenamiento...')
        gt_file = path + '/gt.txt'
        num_lines = sum(1 for line in open(gt_file, 'r'))
        with open(gt_file, 'r') as f:
            for i, line in enumerate(tqdm(f, total=num_lines)):
                image_info = self.line_to_img(line, path)
                if image_info[0] in self.train_images_info.keys():
                    self.train_images_info[image_info[0]].append(image_info[1])
                else:
                    self.train_images_info[image_info[0]] = [image_info[1]]
        self.generate_incorrect_data(path)

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
                trash_regions = self.detector.detect_regions(img)
                if filename in self.train_images_info.keys():
                    signal_regions = self.train_images_info[filename]
                    for trash_region in trash_regions:
                        condition = True
                        for signal_region in signal_regions:
                            if(self.detector.overlap_rectangle(trash_region, signal_region) > 0.1):
                                condition = False
                    if condition:
                        incorrect_data.append(
                            img[trash_region[1]:trash_region[1]+trash_region[3], trash_region[0]:trash_region[0]+trash_region[2]])
                else:
                    for trash_region in trash_regions:
                        incorrect_data.append(
                            img[trash_region[1]:trash_region[1]+trash_region[3], trash_region[0]:trash_region[0]+trash_region[2]])

        self.clasificadores_binarios[SignalType.NO_SEÑAL] = incorrect_data

        # create a folder called debugg, and inside it, save the images for each signal type in self.clasificadores_binarios
    def save_debugg(self):
        if not os.path.exists('debugg'):
            os.makedirs('debugg')

        for signal_type in self.clasificadores_binarios.keys():
            if not os.path.exists('debugg/' + signal_type.name):
                os.makedirs('debugg/' + signal_type.name)
            for i, image in enumerate(self.clasificadores_binarios[signal_type]):
                cv2.imwrite('debugg/' + signal_type.name +
                            '/' + str(i) + '.jpg', image)

    def data_treatment(self):
        print('Aplicando el algoritmo HOG a las imagenes de entrenamiento...')
        for key in tqdm(self.clasificadores_binarios.keys()):
            imagenes_señal = self.clasificadores_binarios[key]
            feature_vector = []
            for img in imagenes_señal:
                hog_result = self.detector.hog(img)
                feature_vector = feature_vector + [hog_result]
            self.clasificadores_binarios[key] = feature_vector

        print('Aplicando reduccion de dimensionalidad a las imagenes de entrenamiento con el algoritmo LDA y generando Clasificadores binarios ...')
        no_signal_data = self.clasificadores_binarios[SignalType.NO_SEÑAL]
        del self.clasificadores_binarios[SignalType.NO_SEÑAL]
        for key in tqdm(self.clasificadores_binarios.keys()):
            lda = LinearDiscriminantAnalysis()
            signal_data = self.clasificadores_binarios[key]
            signal_data = signal_data+no_signal_data
            signal_data = np.array(signal_data, dtype=np.float32)
            aux = np.zeros(len(no_signal_data), dtype=np.int16)
            labels = np.ones(
                len(self.clasificadores_binarios[key]), dtype=np.int16)
            labels = labels*key.value
            labels = np.append(labels, aux)
            labels = labels.astype(np.float32)
            self.clasificadores_binarios[key] = lda.fit(signal_data, labels)

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

    def save_images(self, processed_images: dict):
        '''
        Guarda las imagenes procesadas en el directorio resultado_imgs
        '''
        print('Guardando imagenes procesadas...')
        # create the folder if it does not exist in the same directory as the script
        if not os.path.exists('./resultado_imgs'):
            os.makedirs('./resultado_imgs')
        for name, img_data in tqdm(processed_images.items()):
            cv2.imwrite(f'./resultado_imgs/{name}.jpg', img_data[0])

        print('Guardando informacion de las imagenes procesadas...')
        # create a resultado.txt empty to save the information if it does not exist, if it does exist, it will be overwritten
        if not os.path.exists('resultado.txt'):
            open('resultado.txt', 'w').close()
        # the lines of the resultado.txt file will have the following format:
        # image_name.ppm;x;y;w;h;SignalType;ssim_score
        with open('resultado.txt', 'w') as f:
            for name, img_data in tqdm(processed_images.items()):
                for region in img_data[1]:
                    f.write(
                        f'{name}.jpg;{region[0][0]};{region[0][1]};{region[0][2]};{region[0][3]};{region[1][0].value};{region[1][1]}\n')
