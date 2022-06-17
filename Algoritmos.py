
import matplotlib
from matplotlib import pyplot
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
default_mask_dimension = (32, 32)


delta = 2
max_variation = 0.50
min_area = 65
max_area = 22500


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
            if rect[2] > rect[3]:
                if 1 - variation < rect[3] / rect[2] < 1 + variation:
                    filtered_rects.append(rect)
            else:
                if 1 - variation < rect[2] / rect[3] < 1 + variation:
                    filtered_rects.append(rect)

        return filtered_rects

    def get_IoU(self, rectangle1, rectangle2):
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
        if u == 0 and i != 0:
            return 1.0
        if u == 0:
            return 0.0
        return abs(round(i/u, 2))

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
        # compute the intersection arean
        x1 = max(rectangle1[0], rectangle2[0])
        y1 = max(rectangle1[1], rectangle2[1])
        x2 = min(rectangle1[0] + rectangle1[2], rectangle2[0] + rectangle2[2])
        y2 = min(rectangle1[1] + rectangle1[3], rectangle2[1] + rectangle2[3])
        intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
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

        return rectangle1[2] * rectangle1[3] + rectangle2[2] * rectangle2[3] - self.intersection_area(rectangle1, rectangle2)

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
        # equalize the blue and red channels
        image_copy = image.copy()
        image_copy[:, :, 0] = cv2.equalizeHist(image_copy[:, :, 0])
        image_copy[:, :, 2] = cv2.equalizeHist(image_copy[:, :, 2])
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
        # equilize the histogram of the image
        return bigest_ratios.astype(np.uint8)

    def amplify_single_area(self, rect, factor, shape):
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
        x, y, w, h = rect
        pixels = int(2*min(w, h) * factor)
        if x > pixels:
            x = x - int(pixels/2)
        else:
            x = 1
        if y > pixels:
            y = y - int(pixels/2)
        else:
            y = 1
        if w < shape[1] - pixels:
            w = w + pixels
        else:
            w = shape[1]
        if h < shape[0] - pixels:
            h = h + pixels
        else:
            h = shape[0]

        return (x, y, w, h)

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

            pixels = 2*min(w, h) * factor
            pixels = int(pixels)
            if x > pixels:
                x = x - int(pixels/2)
            else:
                x = 1
            if y > pixels:
                y = y - int(pixels/2)
            else:
                y = 1
            if w < shape[1] - pixels:
                w = w + pixels
            else:
                w = shape[1]
            if h < shape[0] - pixels:
                h = h + pixels
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
        bboxes = self.filter_squares(bboxes, 0.15)
        bboxes = self.amplify_area(bboxes, 0.15, image.shape)
        return bboxes

    # make a method that given a lis of bboxes of the form (x,y,w,h), and a threshold
    # check that the squares of the bboxes dont overlap over the threshold, if they do, remove them from the list

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
        bboxes = sorted(bboxes, key=lambda x: x[2]*x[3], reverse=True)

        copy_list = bboxes.copy()
        for region in bboxes:
            for region2 in bboxes:
                if not self.equal_region(region, region2):
                    if self.get_IoU(region, region2) > threshold and region[2]*region[3] > region2[2]*region2[3]:
                        if region2 in copy_list:
                            copy_list.remove(region2)
        copy_list = list(set(copy_list))
        return copy_list

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
                image, info[0].name+""+str(info[1]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
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

    def multi_class_classifier(self, test_images, clasificadores_binarios):
        tests_images_classified = {}
        print('Aplicando clasificador multiclase...')
        for image in tqdm(test_images.keys()):
            image_copy = test_images[image]
            regions = self.detect_regions(image_copy)
            regions = self.filter_overlapping_squares(regions, 0.50)
            classified_regions = []
            for region in regions:
                x, y, w, h = region
                cropped_image = image_copy[y:y+h, x:x+w]
                if cropped_image.shape[0] != 0 and cropped_image.shape[1] != 0:
                    cropped_image = cv2.resize(cropped_image, (32, 32))
                    hog_result = self.hog(cropped_image)
                    best_match = None
                    best_match_value = 0.0
                    for key in clasificadores_binarios.keys():
                        probability = clasificadores_binarios[key].predict_proba(
                            [hog_result])
                        probability = probability[0][1]
                        probability = np.round(probability, 2)
                        if probability > best_match_value:
                            best_match = key
                            best_match_value = probability
                    if (best_match is not None) and (best_match_value > 0.5):
                        classified_regions.append(
                            (region, (best_match, best_match_value)))
            printed_image = self.print_rectangles(
                image_copy, classified_regions)
            tests_images_classified[image] = (
                printed_image, classified_regions)

        return tests_images_classified
