import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from recomendador_simple import RecomendadorSimple

class RecomendadorIA:
    def __init__(self, modelo_path: str = 'modelos/modelo_ia.joblib'):
        """
        Inicializa el recomendador de IA que combina reglas con aprendizaje automático.
        
        Args:
            modelo_path: Ruta donde se guardará/cargará el modelo entrenado
        """
        self.modelo_path = modelo_path
        self.recomendador_simple = RecomendadorSimple()
        self.modelo = None
        self.label_encoders = {
            'genotipo_c19': LabelEncoder(),
            'genotipo_d6': LabelEncoder(),
            'evaluacion': LabelEncoder()
        }
        
        # Puntuación máxima para cálculo de porcentaje
        self.PUNTUACION_MAXIMA = 1.5
        
        # Cargar o entrenar el modelo
        if os.path.exists(modelo_path):
            self.cargar_modelo()
        else:
            self.entrenar_modelo()
    
    def _preparar_datos_entrenamiento(self) -> pd.DataFrame:
        """
        Prepara el conjunto de datos para entrenar el modelo de IA
        usando el recomendador simple como base.
        
        Returns:
            DataFrame con los datos de entrenamiento
        """
        # Definir genotipos válidos
        genotipos_c19 = ['G/G + C/C', 'A/G + C/C', 'A/A + C/C', 'G/G + C/T', 'G/G + T/T', 'A/G + C/T']
        genotipos_d6 = ['G/G', 'G/A', 'A/A']
        
        # Lista para almacenar los datos
        datos_entrenamiento = []
        
        # Generar combinaciones de genotipos y fármacos
        for genotipo_c19 in genotipos_c19:
            for genotipo_d6 in genotipos_d6:
                try:
                    # Obtener recomendaciones del sistema simple
                    mejores, _ = self.recomendador_simple.recomendar_farmacos(genotipo_c19, genotipo_d6)
                    
                    for rec in mejores:
                        # Extraer características relevantes
                        for gen in ['CYP2C19', 'CYP2D6']:
                            if rec['presente_en'][gen]:
                                predicciones = rec['predicciones'][gen]
                                # Verificar que todas las claves necesarias estén presentes
                                if all(key in predicciones for key in ['evaluacion', 'confianza', 'p_value', 
                                                                      'puntuacion_base', 'factor_p_value']):
                                    datos_entrenamiento.append({
                                        'genotipo_c19': genotipo_c19,
                                        'genotipo_d6': genotipo_d6,
                                        'farmaco': rec['farmaco'],
                                        'gen': gen,
                                        'evaluacion': predicciones['evaluacion'],
                                        'confianza': predicciones['confianza'],
                                        'p_value': predicciones['p_value'],
                                        'puntuacion_base': predicciones['puntuacion_base'],
                                        'factor_p_value': predicciones['factor_p_value'],
                                        'puntuacion_final': rec['puntuacion'],
                                        'porcentaje_exito': rec['porcentaje_exito']
                                    })
                except Exception as e:
                    print(f"Advertencia: Error al procesar genotipos {genotipo_c19}/{genotipo_d6}: {str(e)}")
                    continue
        
        if not datos_entrenamiento:
            raise ValueError("No se pudieron generar datos de entrenamiento válidos")
        
        return pd.DataFrame(datos_entrenamiento)
    
    def _preparar_caracteristicas(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara las características para el entrenamiento o predicción.
        
        Args:
            df: DataFrame con los datos
            
        Returns:
            tuple: (X, y) donde X son las características y y es la variable objetivo
        """
        # Codificar variables categóricas
        X = df.copy()
        for col, encoder in self.label_encoders.items():
            if col in X.columns:
                if not hasattr(encoder, 'classes_'):
                    encoder.fit(X[col])
                X[col] = encoder.transform(X[col])
        
        # Seleccionar características para el modelo
        features = ['genotipo_c19', 'genotipo_d6', 'confianza', 'p_value', 
                   'puntuacion_base', 'factor_p_value']
        X = X[features].values
        y = df['puntuacion_final'].values
        
        return X, y
    
    def entrenar_modelo(self):
        """Entrena el modelo de IA usando los datos del recomendador simple."""
        print("\nPreparando datos de entrenamiento...")
        df = self._preparar_datos_entrenamiento()
        
        print("Entrenando modelo de IA...")
        X, y = self._preparar_caracteristicas(df)
        
        # Dividir datos en entrenamiento y validación
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Crear y entrenar el modelo
        self.modelo = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.modelo.fit(X_train, y_train)
        
        # Evaluar el modelo
        y_pred = self.modelo.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        print(f"\nMétricas del modelo:")
        print(f"MSE: {mse:.4f}")
        print(f"R²: {r2:.4f}")
        
        # Guardar el modelo
        os.makedirs(os.path.dirname(self.modelo_path), exist_ok=True)
        joblib.dump(self.modelo, self.modelo_path)
        print(f"\nModelo guardado en: {self.modelo_path}")
    
    def cargar_modelo(self):
        """Carga el modelo entrenado desde el archivo."""
        try:
            self.modelo = joblib.load(self.modelo_path)
            print(f"Modelo cargado desde: {self.modelo_path}")
        except Exception as e:
            print(f"Error al cargar el modelo: {str(e)}")
            print("Entrenando nuevo modelo...")
            self.entrenar_modelo()
    
    def _predecir_ia(self, genotipo_c19: str, genotipo_d6: str, 
                     farmaco: str, detalles_gen: Dict) -> float:
        """
        Realiza la predicción usando el modelo de IA.
        
        Args:
            genotipo_c19: Genotipo de CYP2C19
            genotipo_d6: Genotipo de CYP2D6
            farmaco: Nombre del fármaco
            detalles_gen: Detalles de la predicción del gen
            
        Returns:
            float: Puntuación predicha por la IA
        """
        if self.modelo is None:
            return 0.0
        
        # Verificar que todas las claves necesarias estén presentes
        claves_requeridas = ['confianza', 'p_value', 'puntuacion_base', 'factor_p_value']
        if not all(key in detalles_gen for key in claves_requeridas):
            print(f"Advertencia: Faltan datos necesarios para la predicción de IA para {farmaco}")
            return detalles_gen.get('puntuacion_base', 0.0)
        
        try:
            # Preparar datos para predicción
            datos = pd.DataFrame([{
                'genotipo_c19': genotipo_c19,
                'genotipo_d6': genotipo_d6,
                'confianza': detalles_gen['confianza'],
                'p_value': detalles_gen['p_value'],
                'puntuacion_base': detalles_gen['puntuacion_base'],
                'factor_p_value': detalles_gen['factor_p_value']
            }])
            
            # Preparar características
            X, _ = self._preparar_caracteristicas(datos)
            
            # Realizar predicción
            return float(self.modelo.predict(X)[0])
        except Exception as e:
            print(f"Error en predicción de IA para {farmaco}: {str(e)}")
            return detalles_gen.get('puntuacion_base', 0.0)
    
    def recomendar_farmacos(self, genotipo_c19: str, genotipo_d6: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Genera recomendaciones combinando el sistema basado en reglas con la IA.
        
        Args:
            genotipo_c19: Genotipo de CYP2C19
            genotipo_d6: Genotipo de CYP2D6
        Returns:
            tuple: (mejores_recomendaciones, peores_recomendaciones)
        """
        # Obtener recomendaciones del sistema simple
        mejores_simple, peores_simple = self.recomendador_simple.recomendar_farmacos(
            genotipo_c19, genotipo_d6, 10000
        )
        # Combinar predicciones
        recomendaciones_combinadas = []
        for rec in mejores_simple + peores_simple:
            predicciones_ia = {}
            for gen in ['CYP2C19', 'CYP2D6']:
                if rec['presente_en'][gen]:
                    predicciones = rec['predicciones'][gen]
                    puntuacion_ia = self._predecir_ia(
                        genotipo_c19, genotipo_d6, rec['farmaco'], predicciones
                    )
                    predicciones_ia[gen] = puntuacion_ia
            if predicciones_ia:
                puntuacion_ia = sum(predicciones_ia.values()) / len(predicciones_ia)
                puntuacion_combinada = 0.7 * puntuacion_ia + 0.3 * rec['puntuacion']
            else:
                puntuacion_combinada = rec['puntuacion']
            rec_combinada = rec.copy()
            rec_combinada['puntuacion'] = puntuacion_combinada
            rec_combinada['porcentaje_exito'] = (puntuacion_combinada / self.PUNTUACION_MAXIMA) * 100
            rec_combinada['predicciones_ia'] = predicciones_ia
            recomendaciones_combinadas.append(rec_combinada)
        recomendaciones_ordenadas = sorted(
            recomendaciones_combinadas,
            key=lambda x: x['puntuacion'],
            reverse=True
        )
        # Ahora devolver todos como mejores y peores (mejores orden descendente, peores ascendente)
        mejores = recomendaciones_ordenadas
        peores = list(reversed(recomendaciones_ordenadas))
        return mejores, peores 