
from cv2 import cvtColor
from more_itertools import sample
import numpy as np
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import PrecisionRecallDisplay
import pandas as pd
import os

from Image import SignalType

default_mask_dimension = (32, 32)
delta = 12
max_variation = 0.75
min_area = 75
max_area = 2250


class Detector:

    def __init__(self):
        self.mser = cv2.MSER_create(
            delta=delta, min_area=min_area, max_area=max_area, max_variation=max_variation)
        self.mser_rb = cv2.MSER_create(delta=2, min_area=75,
                                       max_area=22500, max_variation=0.75)

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
        if rectangle1[0] > rectangle2[0] and rectangle1[1] > rectangle2[1] and rectangle1[0] + rectangle1[2] < rectangle2[0] + rectangle2[2] and rectangle1[1] + rectangle1[3] < rectangle2[1] + rectangle2[3]:
            return 1.0
        if rectangle2[0] > rectangle1[0] and rectangle2[1] > rectangle1[1] and rectangle2[0] + rectangle2[2] < rectangle1[0] + rectangle1[2] and rectangle2[1] + rectangle2[3] < rectangle1[1] + rectangle1[3]:
            return 1.0

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

    def ligth_detect_regions(self, image):
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
        msers, bboxes = self.mser_rb.detectRegions(enchance_image)
        bboxes = self.filter_squares(bboxes, 0.15)
        bboxes = self.amplify_area(bboxes, 0.12, image.shape)
        bboxes = self.filter_overlapping_squares(bboxes, 0.5)
        bboxes = list(set(bboxes))
        return bboxes

    def complete_detect_regions(self, image):
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
        enchance_image = cvtColor(image, cv2.COLOR_BGR2GRAY)
        enchance_image = cv2.equalizeHist(enchance_image)
        msers, bboxes = self.mser.detectRegions(enchance_image)
        enchance_image = self.enhance_blue_red(image)
        msers, secondbboxes = self.mser_rb.detectRegions(enchance_image)
        bboxes = np.append(bboxes, secondbboxes, axis=0)
        bboxes = self.filter_squares(bboxes, 0.15)
        bboxes = self.amplify_area(bboxes, 0.12, image.shape)
        bboxes = self.filter_overlapping_squares(bboxes, 0.5)
        bboxes = list(set(bboxes))
        return bboxes

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

        image = self.resize_img(image)
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

    def multi_class_classifier_LDA_BAYES(self, test_images, clasificadores_binarios, clasification_type):
        tests_images_classified = {}
        print('Aplicando clasificador multiclase...')
        for image in tqdm(test_images.keys()):
            image_copy = test_images[image]
            # regions = self.complete_detect_regions(image_copy)
            regions = self.ligth_detect_regions(image_copy)

            classified_regions = self.clasificador_regiones(
                clasificadores_binarios, image_copy, regions, clasification_type)

            printed_image = self.print_rectangles(
                image_copy, classified_regions)
            tests_images_classified[image] = (
                printed_image, classified_regions)
        return tests_images_classified

    def clasificador_regiones(self, clasificadores_binarios, image_copy, regions, clasification_type):
        classified_regions = []
        for region in regions:
            x, y, w, h = region
            cropped_image = image_copy[y:y+h, x:x+w]
            if cropped_image.shape[0] != 0 and cropped_image.shape[1] != 0:
                cropped_image = self.resize_img(cropped_image)
                if clasification_type == 'HOG_LDA_BAYES':
                    cropped_image = self.hog(cropped_image)
                elif clasification_type == 'GRAY_LDA_BAYES':
                    cropped_image = cv2.cvtColor(
                        cropped_image, cv2.COLOR_BGR2GRAY)
                    cropped_image = cropped_image.flatten()

                best_match, best_match_value = self.find_best_match(
                    clasificadores_binarios, cropped_image)

                if (best_match is not None) and (best_match_value > 0.5):
                    classified_regions.append(
                        (region, (best_match, best_match_value)))

        return classified_regions

    def find_best_match(self, clasificadores_binarios, feature_vector):
        best_match = None
        best_match_value = 0.0
        for key in clasificadores_binarios.keys():
            probability = clasificadores_binarios[key].predict_proba(
                [feature_vector])
            probability = probability[0][1]
            probability = np.round(probability, 2)
            if probability > best_match_value:
                best_match = key
                best_match_value = probability
        return best_match, best_match_value

    def evaluate_classifier(self, validation_set, clasificadores_binarios):
        print('Evaluando clasificador...')
        for _ in tqdm(range(1)):
            # create a folder called "Evaluation" to save the results
            if not os.path.exists("Evaluation"):
                os.makedirs("Evaluation")

            # generacion de datos de validacion
            y_true = self.generate_y_true(validation_set)
            y_pred = self.generate_y_pred(
                validation_set, clasificadores_binarios)
            # creacion de la matriz de confusion
            self.get_confussion_matrix(y_true, y_pred)
            # generacion de tabla con datos estadisticos
            self.get_statics_table(y_true, y_pred)

            self.get_roc_curve(
                validation_set, y_true, y_pred)

    def get_roc_curve(self, validation_set, y_true, y_pred):
        for key in validation_set:
            if key is not SignalType.NO_SEÑAL:
                RocCurveDisplay.from_predictions(
                    y_true=y_true, y_pred=y_pred, pos_label=key.value, name=key.name)
                plt.savefig('Evaluation/roc_curve'+key.name+'.png')
                plt.close()

    def get_statics_table(self, y_true, y_pred):
        accuracy_score_value = accuracy_score(
            y_true, y_pred, normalize=True)
        precision_score_value = precision_score(
            y_true, y_pred, labels=[1, 2, 3, 4, 5, 6], average='micro')
        recall_score_value = recall_score(
            y_true, y_pred, labels=[1, 2, 3, 4, 5, 6], average='micro')
        fi_score_value = f1_score(y_true, y_pred, labels=[
            1, 2, 3, 4, 5, 6], average='micro')
        table = pd.DataFrame({'Metricas': ['Accuracy', 'Precision', 'Recall', 'F1_Score'],   'Valores': [
            accuracy_score_value, precision_score_value, recall_score_value, fi_score_value]})
        plt.figure(figsize=(10, 5))
        plt.title('Evaluación del clasificador')
        plt.axis('off')
        plt.table(cellText=table.values,
                  colLabels=table.columns, loc='center')
        plt.savefig('Evaluation/datos_estadisticos.png')
        plt.close()

    def get_confussion_matrix(self, y_true, y_pred):
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
        plt.title('Matriz de confusión')
        plt.savefig('Evaluation/matriz_confusion.png')
        plt.close()

    def generate_y_pred(self, validation_set, clasificadores_binarios):
        y_pred = []
        for key in validation_set.keys():
            for element in validation_set[key]:
                best_match = 0
                for clasificador in clasificadores_binarios.keys():
                    match_class = clasificadores_binarios[clasificador].predict(
                        [element])[0]
                    if match_class != 0:
                        best_match = match_class
                        break
                y_pred.append(best_match)
        return y_pred

    def generate_y_true(self, validation_set):
        y_true = []
        for key in validation_set.keys():
            for _ in validation_set[key]:
                y_true.append(key.value)
        return y_true
