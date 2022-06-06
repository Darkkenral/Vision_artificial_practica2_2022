from distutils.log import info
from typing import final
from cv2 import cvtColor, mean
from skimage.metrics import structural_similarity as ssim
import numpy as np
from tqdm import tqdm
import cv2
from matplotlib import contour, image, pyplot

default_mask_dimension = (32, 32)


delta = 2
max_variation = 0.75
min_area = 150
max_area = 28900


class Detector:

    def __init__(self):
        self.mser = cv2.MSER_create(
            delta=delta, min_area=min_area, max_area=max_area, max_variation=max_variation)

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

    def filter_squares(self, rects, variation):
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
        if rectangle1[0] > rectangle2[0] and rectangle1[1] > rectangle2[1] and rectangle1[0] + rectangle1[2] < rectangle2[0] + rectangle2[2] and rectangle1[1] + rectangle1[3] < rectangle2[1] + rectangle2[3]:
            return 1.0
        if rectangle2[0] > rectangle1[0] and rectangle2[1] > rectangle1[1] and rectangle2[0] + rectangle2[2] < rectangle1[0] + rectangle1[2] and rectangle2[1] + rectangle2[3] < rectangle1[1] + rectangle1[3]:
            return 1.0

        i = self.intersection_area(rectangle1, rectangle2)
        u = self.union_area(rectangle1, rectangle2)
        if u == 0:
            return 0.0
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
        # compute the intersection area
        intersection_area = (
            min(rectangle1[0] + rectangle1[2], rectangle2[0] + rectangle2[2]) - max(rectangle1[0], rectangle2[0])) * (
                min(rectangle1[1] + rectangle1[3], rectangle2[1] + rectangle2[3]) - max(rectangle1[1], rectangle2[1]))
        return intersection_area

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
        # compute the union area
        union_area = (
            rectangle1[2] * rectangle1[3] + rectangle2[2] * rectangle2[3] - self.intersection_area(rectangle1, rectangle2))
        return union_area

    def enhance_blue_red(self, image):
        '''
        Aumenta el contraste de la imagen en el espacio de color BGR

        Parameters
        ----------
        image : Imagen
            Imagen a aumentar el contraste

        Returns
        -------
        Imagen
            Imagen con el contraste aumentado
        '''
        image_copy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
        image_copy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)
        b, g, r = cv2.split(image_copy)
        b_chanel = np.array(b, dtype=np.int16)
        g_chanel = np.array(g, dtype=np.int16)
        r_chanel = np.array(r, dtype=np.int16)
        sum_chanels = b_chanel + g_chanel + r_chanel
        sum_chanels[sum_chanels == 0] = 1
        red_ratio = r_chanel / sum_chanels
        blue_ratio = b_chanel / sum_chanels
        red_ratio = red_ratio * 100.0
        blue_ratio = blue_ratio * 100.0
        bigest_ratios = np.array(np.maximum(
            red_ratio, blue_ratio), dtype=np.int16)
        bigest_ratios[bigest_ratios > 255] = 255
        return bigest_ratios.astype(np.uint8)

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
        return_list = []
        for rect in rects:
            x, y, w, h = rect
            if x > factor:
                x = x - int(factor/2)
            else:
                x = 1
            if y > factor:
                y = y - int(factor/2)
            else:
                y = 1
            if w < shape[1] - factor:
                w = w + factor
            else:
                w = shape[1]
            if h < shape[0] - factor:
                h = h + factor
            else:
                h = shape[0]

            return_list.append((x, y, w, h))

        return return_list

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
        enchance_image = self.enhance_blue_red(image)
        msers, bboxes = self.mser.detectRegions(enchance_image)
        bboxes = self.filter_squares(bboxes, 0.25)
        bboxes = self.amplify_area(bboxes, 20, image.shape)
        # turn into set and then into list to remove duplicates
        bboxes = list(set(bboxes))
        return bboxes
    # make a method that given a lis of bboxes of the form (x,y,w,h), and a threshold
    # check that the squares of the bboxes dont overlap too much with each other

    def filter_overlapping_squares(self, bboxes, threshold):
        '''
        Filtra los rectángulos de una lista de rectángulos

        Parameters
        ----------
        bboxes : list
            Lista de rectángulos
        threshold : float
            Umbral de intersección

        Returns
        -------
        list
            Lista de rectángulos filtrados
        '''
        return_list = []
        bboxes = sorted(bboxes, key=lambda x: x[2]*x[3], reverse=True)
        for i in range(len(bboxes)):
            condition = True
            for j in range(i + 1, len(bboxes)):
                if self.overlap_rectangle(bboxes[i], bboxes[j]) > threshold:
                    condition = False
                    break
            if condition:
                return_list.append(bboxes[i])
        return return_list

    def equal_region(self, region1, region2):
        '''
        Compara dos regiones de interes

        Parameters
        ----------
        region1 : bunding box
            Primera region de interes
        region2 : bunding box
            Segunda region de interes
        '''
        x1, y1, w1, h1 = region1
        x2, y2, w2, h2 = region2
        return (x1 == x2) and (y1 == y2) and (w1 == w2) and (h1 == h2)

    def get_biggest(self, regions):
        '''
        Obtiene el rectángulo con mayor área de una lista de rectángulos

        Parameters
        ----------
        regions : list
            Lista de rectángulos

        Returns
        -------
        bunding box
            Rectángulo con mayor área
        '''
        biggest_region = regions[0]
        for region in regions:
            if region[2] * region[3] > biggest_region[2] * biggest_region[3]:
                biggest_region = region
        return biggest_region

    def print_rectangles(self, image, save_regions):
        for region, info in save_regions:
            x, y, w, h = region
            image = cv2.rectangle(
                image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            image = cv2.putText(
                image, info[0].name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return image

    def hog(self, image):

        image = cv2.resize(image, (32, 32))
        # Parametros HOG
        winSize = (32, 32)
        blockSize = (16, 16)
        blockStride = (8, 8)
        cellSize = (8, 8)
        nbins = 9
        derivAperture = 1
        winSigma = 4.
        histogramNormType = 0
        L2HysThreshold = 2.0000000000000001e-01
        gammaCorrection = 0
        nlevels = 64
        # HOG
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture,
                                winSigma,   histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
        # Generando los vectores de caracteristicas
        feature_vector = hog.compute(image)
        # print the dtype of the feature vector
        feature_vector = list(feature_vector)
        return feature_vector
