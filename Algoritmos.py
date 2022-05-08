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

    def gaussian_blurr(self, image, gaussian_mask=(5, 5), alpha=0):
        '''
        Aplica un filtro gaussiano a una imagen

        Parameters
        ----------
        image : Imagen
            Imagen a la que se le aplicará el filtro
        gaussian_mask : tuple, optional
            mascara gausiana,por defecto (5, 5)
        alpha : int, optional
            valor de aplicación del filtro, por defecto 0

        Returns
        -------
        Imagen
            Imagen con el filtro aplicado
        '''
        return cv2.GaussianBlur(image, gaussian_mask, alpha)

    def color_mask(self, image, min, max):
        '''
        Crea una máscara de una imagen con una máscara de color

        Parameters
        ----------
        image : Imagen
            Imagen a la que se le aplicará la máscara
        min : tuple
            Umbral inferior de la máscara
        max : tuple
            Umbral superior de la máscara

        Returns
        -------
        Imagen
            Imagen con la máscara aplicada
        '''
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        umbral = cv2.inRange(image, min, max)
        extraido = cv2.bitwise_and(image, image, mask=umbral)
        image = cv2.cvtColor(extraido, cv2.COLOR_RGB2GRAY)

        ret, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
        ret, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.resize(binary,
                            default_mask_dimension,
                            interpolation=cv2.INTER_AREA)
        return binary

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

    def color_thresholding(self, image):
        '''
        Aplica un filtro de color a una imagen

        Parameters
        ----------
        image : Imagen
            Imagen a la que se le aplicará el filtro

        Returns
        -------
        Imagen
            Imagen con el filtro aplicado
        '''
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        umbral_rojo = cv2.inRange(image, red_min, red_max)
        umbral_blue = cv2.inRange(image, blue_min, blue_max)
        umbral = cv2.bitwise_or(umbral_rojo, umbral_blue)
        extraido = cv2.bitwise_and(image, image, mask=umbral)
        image = cv2.cvtColor(extraido, cv2.COLOR_RGB2GRAY)
        ret, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY)
        kernel = np.ones((4, 4), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return binary

    def brightness_contrast(self, image, brightness, contrast):
        '''
        Aplica una ajuste de brillo y contraste a una imagen
        '''
        alpha = 1.0 + (contrast / 100.0)
        beta = brightness - (brightness * (contrast / 100.0))
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted

    def otsu_thresholding(self, image):
        '''
        Aplica una umbralización Otsu a una imagen

        Parameters
        ----------
        image : Imagen
            Imagen a la que se le aplicará el filtro

        Returns
        -------
        Imagen
            mascara binaria de la imagen con la umbralización Otsu
        '''
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        ret, binary = cv2.threshold(gray, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def equalize_histogram(self, image):
        '''
        Ajusta el histograma de una imagen en color
        '''
        ycbr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        ycbr[:, :, 0] = cv2.equalizeHist(ycbr[:, :, 0])
        return cv2.cvtColor(ycbr, cv2.COLOR_YCrCb2BGR)

    def detect_circles(self, image):
        '''
        Detecta los círculos en una imagen aplicando la transformada de Hough

        Parameters
        ----------
        image : Imagenq
            Imagen a la que se le aplicará la detección de círculos

        Returns
        -------
        lista
            Lista de regiones detectadas con los círculos dentro de ellas
        '''

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 20,
                                   param1=130, param2=100, minRadius=5,
                                   maxRadius=200)
        regions_detected = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                x_c, y_c, r = i[0], i[1], i[2]
                x_c = x_c.astype(int)
                y_c = y_c.astype(int)
                r = r.astype(int)
                x = int(np.subtract(x_c, r))
                y = int(np.subtract(y_c, r))
                w = int(np.multiply(r, 2))
                h = int(np.multiply(r, 2))
                if x > 10:
                    x = x - 10
                else:
                    x = 0
                if y > 10:
                    y = y - 10
                else:
                    y = 0
                if w < image.shape[1] - 10:
                    w = w + 10
                else:
                    w = image.shape[1]
                if h < image.shape[0] - 10:
                    h = h + 10

                rectangle = (x, y, w, h)

                regions_detected.append(rectangle)

        return regions_detected

    def detect_triangles(self, image):
        '''
        Detecta los triángulos en una imagen aplicando deteccion de contornos y  aproximacion de polígonos
        Parameters
        ----------
        image : Imagen
            Imagen a la que se le aplicará la detección de triángulos

        Returns
        -------
        lista
            Lista de coordenadas de los triángulos detectados
        '''
        image_equalized = self.equalize_histogram(image)
        triangles = self.detect_geometric_forms(image_equalized, 3)
        return triangles

    def detect_hexagons(self, image):
        '''
        Detecta los hexagonos en una imagen aplicando deteccion de contornos y  aproximacion de polígonos
        Parameters
        ----------
        image : Imagen
            Imagen a la que se le aplicará la detección de hexagonos

        Returns
        -------
        lista
            Lista de coordenadas de los hexagonos detectados
        '''
        image_equalized = self.equalize_histogram(image)
        hexagons = self.detect_geometric_forms(image_equalized, 6)
        return hexagons

    def detect_geometric_forms(self, image, corners):
        '''
        Detecta las figuras geometricas en una imagen aplicando deteccion de contornos y  aproximacion de polígonos y selecciona aquellas con un numero de esquinas similar al indicado
        Parameters
        ----------
        image : Imagen
            Imagen a la que se le aplicará la detección de figuras geometricas

        Returns
        -------
        lista
            Lista de coordenadas de las figuras geometricas detectadas
        '''
        geometric_forms = []
        binary_image = self.color_thresholding(image)
        contours = cv2.findContours(
            binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        for cnt in contours:
            figure = cv2.approxPolyDP(
                cnt, 0.2 * cv2.arcLength(cnt, True), True)
            if len(figure) == corners:
                x, y, w, h = cv2.boundingRect(cnt)
                if x > 10:
                    x = x - 10
                else:
                    x = 1
                if y > 10:
                    y = y - 10
                else:
                    y = 1
                if w < image.shape[1] - 10:
                    w = w + 10
                else:
                    w = image.shape[1]
                if h < image.shape[0] - 10:
                    h = h + 10
                geometric_forms.append((x, y, w, h))
        return geometric_forms

    def indice_similitud_estructural(self, imageA, imageB):
        '''
        Calcula el SSIM entre dos imagenes

        Parameters
        ----------
        imageA : Imagen
            Primera imagen
        imageB : Imagen
            Segunda imagen

        Returns
        -------
        float
            Valor de SSIM entre las dos imagenes
        '''
        return ssim(imageA, imageB)

    def compare_masks(self, mask1, mask2):
        '''
        Compara dos máscaras binarias y devuelve una máscara con los elementos que están en ambas máscaras
        Parameters
        ----------
        mask1 : _type_
            _description_
        mask2 : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        '''
        mask1 = cv2.resize(
            mask1, (default_mask_dimension[0], default_mask_dimension[1]))
        mask2 = cv2.resize(
            mask2, (default_mask_dimension[0], default_mask_dimension[1]))
        similitud = self.indice_similitud_estructural(mask1, mask2)
        return similitud

    def find_best_match(self, mask, masks):
        '''
        Busca la mejor coincidencia entre una máscara y una lista de máscaras

        Parameters
        ----------
        mask : Imagen
            Máscara a comparar
        masks : lista
            Lista de máscaras

        Returns
        -------
        tuple
            Tupla con el nombre con la mejor coincidencia ,el valor de la coincidencia y el error cuadratico medio para la coincidencia
        '''
        best_match = None
        best_similitude_value = 0
        for key in masks:
            simil = self.compare_masks(mask, masks[key])
            if simil > best_similitude_value:
                best_similitude_value = simil
                best_match = key
        return (best_match, best_similitude_value)

    def form_detector(self, test_images: dict, trained_masks: dict):
        '''
        Detecta las figuras geometricas en una imagen aplicando deteccion de contornos y  aproximacion de polígonos  asi como aplicando la transformada de Hough
        Parameters
        ----------
        test_images : dict
            Diccionario con las imagenes de test

        Returns[0]
        -------
        dict
            Diccionario con las imagenes de test con las figuras detectadas y su informacion
        '''
        print("Aplicando la deteccion de forma a las imagenes de test")
        image_processed = {}
        for imagen in tqdm(test_images.keys()):
            regions = []
            regions.extend(self.detect_circles(test_images[imagen]))
            regions.extend(self.detect_triangles(test_images[imagen]))
            regions.extend(self.detect_hexagons(test_images[imagen]))
            save_regions = []
            for region in regions:
                x, y, w, h = region
                # extract the region of interest in a bgr image
                copia_imagen = test_images[imagen]
                roi = copia_imagen[y:y + h, x:x + w, :]
                mask_red = self.color_mask(roi, red_min, red_max)
                mask_blue = self.color_mask(roi, blue_min, blue_max)
                information_red = self.find_best_match(
                    mask_red, trained_masks)
                information_blue = self.find_best_match(
                    mask_blue, trained_masks)
                information_final = []
                if information_red[1] > information_blue[1]:
                    information_final = information_red
                else:
                    information_final = information_blue
                if information_final[1] > 0.47:

                    save_regions.append((region, information_final))
            image_processed[imagen] = (test_images[imagen], save_regions)

        return image_processed

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

    # Calculates score of mask from image against model signal type mask
    def score(self, mask_image, sig_mask):
        # Number of 255s in template mask
        total255s = np.count_nonzero(sig_mask == 255)
        # Number of 0s in template mask
        total0s = np.count_nonzero(sig_mask == 0)

        # Inverse of template mask
        sig_mask_inv = cv2.bitwise_not(sig_mask)
        # Inverse of mask image
        mask_image_inv = cv2.bitwise_not(mask_image)

        # Combine masks to see where 255s match
        combinedmask_255s = cv2.bitwise_and(
            mask_image, mask_image, mask=sig_mask)
        # As there is no NOT AND, we use equivalent NOT A OR NOT B
        combinedmask_0s = cv2.bitwise_or(
            mask_image_inv, mask_image_inv, mask=sig_mask_inv)

        matching255s = np.count_nonzero(
            combinedmask_255s == 255)                               # Count 255s

        # Count 0s (255s in inverted mask)
        matching0s = np.count_nonzero(combinedmask_0s == 255)

        result = ((matching255s / total255s) -
                  ((total0s - matching0s) / total0s))
        return result

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

    def mean_score(self, mask1, mask2):
        suma = 0
        for i in range(len(mask1)):
            for j in range(len(mask1[i])):
                if mask1[i][j] == mask2[i][j]:
                    suma += 1

        return suma / (default_mask_dimension[0]*default_mask_dimension[1])
