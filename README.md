# Instrucciones de uso

En este archivo se proporciona informacion de los pasos a seguir para la instalacion de dependencias y la ejecucion de script.

---

### Instalacion de dependencias

Para la instalacion de dependencias se proporciona un fichero "requierments.txt", el cual puede ser instalado con pip de manera automatica, con todas las dependencias incluidas.

```
pip install -r requirements.txt
```

---

### EJecucion del codigo.

Para realizar la ejecucion del codigo es necesario indicar tanto el detector que se va a implementar como el directorio de las imagenes de entrenamiento asi como el directorio de las imagenes a las cuales se les va a aplicar la deteccion.

Se podra invocar con el siguiente comando desde la carpeta en la que se encuentre el

script "main.py"

```
python3 main.py --train_path="Directorio de las imagenes de entrenamiento" --test_path="Directorio de las imagenes a clasificar" --classifier="Tipo de clasificador"
```

En esta practica el clasificador puede tomar los siguientes tres valores 

- HOG_LDA_BAYES

-  GRAY_LDA_BAYES

-  RGB_LDA_BAYES

-  CANY_LDA_BAYES

-  HOG_PCA_KNN

-  GRAY_PCA_KNN

-  RGB_PCA_KNN 

- CANY_PCA_KNN



### Salida

La imagenes ya procesadas seran guardadas en un directorio llamado "[resultado_imgs](./resultado_imgs)", dicho directorio almacenara tanto las imagenes de test procesadas con las detecciones pintadas en ella como el archivo "[gt.txt](./resultado_imgs/gt.txt)" que almacenara las detecciones y su grado de precision


