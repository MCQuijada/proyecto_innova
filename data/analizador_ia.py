import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class AnalizadorIA:
    def __init__(self):
        self.datos = None
        self.scaler = StandardScaler()
        self.modelo = None
        
    def cargar_datos_procesados(self, datos):
        """
        Carga los datos procesados para análisis
        Args:
            datos (DataFrame): Datos procesados
        """
        self.datos = datos
        print(f"\nDatos cargados para análisis:")
        print(f"Número de filas: {len(self.datos)}")
        print(f"Número de columnas: {len(self.datos.columns)}")
        
    def analizar_conjuntos_datos(self):
        """
        Realiza un análisis inicial de los datos
        """
        if self.datos is None:
            print("Error: No hay datos cargados para analizar")
            return None
            
        try:
            print("\n=== ANÁLISIS INICIAL DE DATOS ===")
            
            # Mostrar información general
            print("\nInformación general del dataset:")
            print(self.datos.info())
            
            # Mostrar estadísticas descriptivas
            print("\nEstadísticas descriptivas:")
            print(self.datos.describe())
            
            # Análisis de correlaciones (si hay columnas numéricas)
            columnas_numericas = self.datos.select_dtypes(include=[np.number]).columns
            if len(columnas_numericas) > 1:
                print("\nMatriz de correlaciones:")
                correlaciones = self.datos[columnas_numericas].corr()
                print(correlaciones)
            
            # Análisis de valores únicos por columna
            print("\nValores únicos por columna:")
            for columna in self.datos.columns:
                n_unicos = self.datos[columna].nunique()
                print(f"{columna}: {n_unicos} valores únicos")
                
                # Si hay pocos valores únicos, mostrarlos
                if n_unicos < 10:
                    print(f"Valores en {columna}:")
                    print(self.datos[columna].value_counts())
            
            return True
            
        except Exception as e:
            print(f"Error al analizar los datos: {str(e)}")
            return None
            
    def preprocesar_datos(self, columna_objetivo=None):
        """
        Preprocesa los datos para el modelo
        Args:
            columna_objetivo (str): Nombre de la columna objetivo para el modelo
        Returns:
            tuple: (X_train, X_test, y_train, y_test) si hay columna objetivo
                   (X_train, X_test) si no hay columna objetivo
        """
        if self.datos is None:
            print("Error: No hay datos cargados para preprocesar")
            return None
            
        try:
            # Separar características y objetivo si se especifica
            if columna_objetivo and columna_objetivo in self.datos.columns:
                X = self.datos.drop(columns=[columna_objetivo])
                y = self.datos[columna_objetivo]
                
                # Convertir variables categóricas a numéricas
                X = pd.get_dummies(X)
                
                # Escalar las características
                X_scaled = self.scaler.fit_transform(X)
                
                # Dividir en conjuntos de entrenamiento y prueba
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42
                )
                
                print("\nDatos preprocesados:")
                print(f"Dimensiones de entrenamiento: {X_train.shape}")
                print(f"Dimensiones de prueba: {X_test.shape}")
                
                return X_train, X_test, y_train, y_test
            else:
                # Si no hay columna objetivo, solo preprocesar características
                X = pd.get_dummies(self.datos)
                X_scaled = self.scaler.fit_transform(X)
                
                # Dividir en conjuntos de entrenamiento y prueba
                X_train, X_test = train_test_split(
                    X_scaled, test_size=0.2, random_state=42
                )
                
                print("\nDatos preprocesados:")
                print(f"Dimensiones de entrenamiento: {X_train.shape}")
                print(f"Dimensiones de prueba: {X_test.shape}")
                
                return X_train, X_test
                
        except Exception as e:
            print(f"Error al preprocesar los datos: {str(e)}")
            return None
            
    def entrenar_modelo(self, X_train, y_train=None):
        """
        Entrena el modelo con los datos preprocesados
        Args:
            X_train: Datos de entrenamiento
            y_train: Etiquetas de entrenamiento (opcional)
        """
        # Este método se implementará según el tipo de modelo que se quiera usar
        pass
        
    def predecir(self, X):
        """
        Realiza predicciones con el modelo entrenado
        Args:
            X: Datos para predecir
        Returns:
            array: Predicciones
        """
        if self.modelo is None:
            print("Error: No hay modelo entrenado")
            return None
            
        try:
            return self.modelo.predict(X)
        except Exception as e:
            print(f"Error al realizar predicciones: {str(e)}")
            return None
            
    def visualizar_resultados(self, y_true=None, y_pred=None):
        """
        Visualiza los resultados del modelo
        Args:
            y_true: Valores reales
            y_pred: Valores predichos
        """
        # Este método se implementará según las necesidades de visualización
        pass 