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
        bboxes = self.amplify_area(bboxes, 25, image.shape)
        return bboxes

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
        return feature_vector
