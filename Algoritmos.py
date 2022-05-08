from cgi import test
import cmath
from email.policy import default
from platform import python_branch
from re import X
from typing import final
from skimage.metrics import structural_similarity as ssim
from cv2 import RANSAC, GaussianBlur, boundingRect, rectangle, resize
import numpy as np
from pytz import country_timezones
from tqdm import tqdm
from Image import Image
import cv2
from matplotlib import contour, image, pyplot

default_mask_dimension = (25, 25)

red_min = (0, 70, 50)
red_max = (10, 255, 255)
blue_min = (100, 100, 100)
blue_max = (255, 255, 150)

delta = 2
max_variation = 0.75
min_area = 150
max_area = 28900


class Detector:

    def __init__(self):
        self.mser = cv2.MSER_create(
            delta=delta, min_area=min_area, max_area=max_area, max_variation=max_variation)

    def average_image(self, image_list: list):
        '''
        Calcula la imagen media de una lista de imágenes

        Parameters
        ----------
        image_list : list
            Lista de imágenes

        Returns
        -------
        Image
            Imagen media
        '''
        avg_image = image_list[0].image
        for i in range(len(image_list)):
            if i == 0:
                pass
            else:
                alpha = 1.0 / (i + 1)
                beta = 1.0 - alpha
                avg_image = cv2.addWeighted(image_list[i].image, alpha,
                                            avg_image, beta, 0.0)
        return avg_image


    def crop_img(self, image, bbox):
        '''
        Extrae una zona de una imagen mas grande

        Parameters
        ----------
        image : Imagen
            Imagen a la que se le extraerá la zona
        bbox : tuple
            Coordenadas de la zona a extraer

        Returns
        -------
        Imagen
            Imagen con la zona extraida
        '''
        return image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2], :]

    def resize_img(self, image):
        '''
        Redimensiona una imagen a la variable global

        Parameters
        ----------
        image : Imagen
            Imagen a redimensionar

        Returns
        -------
        Imagen
            Imagen redimensionada
        '''
        return cv2.resize(image, default_mask_dimension, interpolation=cv2.INTER_AREA)

    # MSER functions

    def filter_squares(self, rects: list, variation):
        '''
        Filtra una lista de rectángulos según una variación máxima

        Parameters
        ----------
        rects : list
            Lista de rectángulos
        variation : float
            Variación máxima

        Returns
        -------
        list
            Lista de rectángulos filtrada
        '''
        filtered_rects = []
        for rect in rects:
            if 1 - variation < rect[2] / rect[3] < 1 + variation:
                filtered_rects.append(rect)
        return filtered_rects

    def overlap_rectangle(self, rectangle1, rectangle2):
        '''
        Calcula el porcentaje de intersección entre dos rectángulos

        Parameters
        ----------
        rectangle1 : bounding box
            Rectángulo 1
        rectangle2 : bounding box
            Rectángulo 2

        Returns
        -------
        float
            Porcentaje de intersección
        '''
        i = self.intersection_area(rectangle1, rectangle2)
        u = self.union_area(rectangle1, rectangle2)
        return i / u

    def intersection_area(self, rectangle1, rectangle2):
        '''
        Calcula el área de intersección de dos rectángulos

        Parameters
        ----------
        rectangle1 : tuple
            Rectángulo 1
        rectangle2 : tuple
            Rectángulo 2

        Returns
        -------
        int
            Área de intersección
        '''
        x1, y1, w1, h1 = rectangle1
        x3, y3, w2, h2 = rectangle2
        x2 = x1 + w1
        y2 = y1 + h1
        x4 = x3 + w2
        y4 = y3 + h2

        x5 = max(x1, x3)
        y5 = max(y1, y3)

        x6 = min(x2, x4)
        y6 = min(y2, y4)

        if x5 > x6 or y5 > y6:
            return 0
        else:
            return (x6-x5) * (y6-y5)

    def union_area(self, rectangle1, rectangle2):
        '''
        Calcula el área de la unión de dos rectángulos

        Parameters
        ----------
        rectangle1 : bunding box
            Primer rectángulo
        rectangle2 : bunding box
            Segundo rectángulo

        Returns
        -------
        int
            Área de la unión
        '''
        x1, y1, w1, h1 = rectangle1
        x3, y3, w2, h2 = rectangle2
        return (w1 * h1) + (w2 * h2) - self.intersection_area(rectangle1, rectangle2)

    def amplify_area(self, rects, factor, shape):
        '''
        Amplía el área de un rectángulo

        Parameters
        ----------
        rects : list
            Lista de rectángulos
        factor : int
            Factor de ampliación
        shape : tuple
            Tamaño de la imagen

        Returns
        -------
        list
            Lista de rectángulos ampliados
        '''

        for rect in rects:
            x, y, w, h = rect
            if x > factor:
                x = x - factor
            else:
                x = 0
            if y > factor:
                y = y - factor
            else:
                y = 0
            if w < shape[1] - factor:
                w = w + factor
            else:
                w = shape[1]
            if h < shape[0] - factor:
                h = h + factor

            rect = (x, y, w, h)
        return rects

    def detect_regions(self, image):
        '''
        Detecta las regiones de interes de una imagen aplicando el algoritmo MSER

        Parameters
        ----------
        image : Imagen
            Imagen a la que se le aplicará el algoritmo MSER

        Returnsimage.shape
        -------
        list
            Lista de regiones de interes
        '''
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        msers, bboxes = self.mser.detectRegions(gray)
        bboxes = self.filter_squares(bboxes, 0.3)
        bboxes = self.amplify_area(bboxes, 20, image.shape)
        return bboxes

    def mser_detector(self, test_images: dict, trained_masks: dict):
        '''
        Detecta las figuras geometricas en una imagen aplicando el algoritmo MSER

        Parameters
        ----------
        test_images : dict
            Diccionario con las imagenes de test
        trained_masks : dict
            Diccionario con las mascaras de entrenamiento

        Returns
        -------
        dict
            Diccionario con las imagenes de test con las figuras detectadas y su informacion
        '''
        image_processed = {}
        print('Generando regiones del mser....')
        for img in tqdm(test_images.keys()):
            save_regions = []
            boxes = self.detect_regions(test_images[img])
            for box in boxes:
                x, y, w, h = box
                crop_img = test_images[img][y:y + h, x:x + w]
                resize_img = self.resize_img(crop_img)
                blue_mask = self.color_mask(resize_img, blue_min, blue_max)
                red_mask = self.color_mask(resize_img, red_min, red_max)
                information_red = self.find_best_match(
                    red_mask, trained_masks)
                information_blue = self.find_best_match(
                    blue_mask, trained_masks)

                if information_red[1] > information_blue[1]:
                    information_final = information_red
                else:
                    information_final = information_blue
                if information_final[1] > 0.48:
                    save_regions.append((box, information_final))

            save_regions = self.filter_overlapping_squares(save_regions, 0.5)
            self.print_rectangles(save_regions, test_images[img])
            image_processed[img] = (test_images[img], save_regions)
        return image_processed

    def filter(self, save_regions, region, area):
        '''
        Filtra una lista de rectángulos según una área mínima

        Parameters
        ----------
        rects : list
            Lista de rectángulos
        rect :  tuple
            Rectángulo
        area :  int
            Área mínima

        Returns
        -------
        list
            Lista de rectángulos filtrada
        '''
        for r in save_regions:
            if region is not r:
                o = self.overlap_rectangle(region[0], r[0])
                if o > area and region[1][1] > r[1][1]:
                    return False
        return True

    def filter_overlapping_squares(self, save_regions, area_percent_overlapped):
        '''
        Filtra una lista de rectángulos según una área mínima

        Parameters
        ----------
        rects : list
            Lista de rectángulos
        area_percent_overlapped : float
            Porcentaje de área que debe tener un rectángulo para que no se elimine

        Returns
        -------
        list
            Lista de rectángulos filtrada
        '''

        save_regions = [region for region in save_regions if self.filter(
            save_regions, region, area_percent_overlapped)]
        return save_regions

    def print_rectangles(self, save_regions, image):
        for region, info in save_regions:
            x, y, w, h = region
            image = cv2.rectangle(
                image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            image = cv2.putText(
                image, info[0].name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
