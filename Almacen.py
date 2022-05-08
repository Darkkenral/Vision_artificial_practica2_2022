from ast import If
import os
import cv2
from matplotlib import image, pyplot
from Image import Image
from Image import SignalType
from tqdm import tqdm
from Algoritmos import *

default_mask_dimension = (60, 60)


class Warehouse:

    def __init__(self):
        self.train_images = {SignalType.PROHIBIDO: [], SignalType.PELIGRO: [], SignalType.STOP: [
        ], SignalType.DIRECCION_PROHIBIDA: [], SignalType.CEDA_EL_PASO: [], SignalType.DIRECCION_OBLIGATORIA: []}
        self.test_images = {}
        self.mser_rectangles = {}
        self.solution = {}
        self.final_images = {}
        #No es necesario, una vez generada la imagen total se puede crear la mascara de una vez
        # No es necesario, una vez generada la imagen total se puede crear la mascara de una vez
        self.average_images = {}
        self.masks = {}
        # Que hace el objeto detector en la clase de entrenamiento?
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
        cropped_img = master_image[miny:maxy, minx:maxx]
        resized_img = cv2.resize(cropped_img, default_mask_dimension, interpolation = cv2.INTER_AREA)
        img = Image(int(args[5]), resized_img)
        return img

    def load_train_images(self, path):
        '''
        Lee el archvio gt.txt, extrae de cada imagen las regiones de interes,clasifica cada señal de cada region y  las almacena en el diccionario train_images
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
                img = self.line_to_img(line, path)
                if img.signal_type in self.train_images:
                    self.train_images[img.signal_type].append(img)
    
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

    
    


    def generate_average_images(self):
        '''
        Calcula la imagen media para cada una de las listas de señales almacenadas en el dicionario train_images
        '''
        print('Generando las imagenes medias...')
        for signal_type in tqdm(SignalType, total=len(SignalType)):
            self.average_images[signal_type] = self.detector.average_image(
                self.train_images[signal_type])

    def generate_color_masks(self):
        '''
        Genera una mascara de color para un rango de color especifico para cada una de las imagenes medias.
        '''
        print('Generando las mascara de las imagen media...')
        for signal_type in tqdm(SignalType, total=len(SignalType)):
            if signal_type != SignalType.DIRECCION_OBLIGATORIA:
                self.masks[signal_type] = self.detector.color_mask(
                    self.average_images[signal_type], red_min, red_max)
            else:
                self.masks[SignalType.DIRECCION_OBLIGATORIA] = self.detector.color_mask(
                    self.average_images[SignalType.DIRECCION_OBLIGATORIA], blue_min, blue_max)

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
        # create the folder if it does not exist
        if not os.path.exists('resultado_imgs'):
            os.makedirs('resultado_imgs')
        for name, img_data in tqdm(processed_images.items()):
            cv2.imwrite(f'resultado_imgs/{name}.jpg', img_data[0])

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
