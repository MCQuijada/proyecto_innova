import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from typing import Dict, List
import joblib
import os

class RecomendadorIA:
    def __init__(self, modelo_path: str = 'modelos/'):
        """
        Inicializa el recomendador basado en IA.
        Args:
            modelo_path: Directorio donde se guardarán los modelos entrenados
        """
        self.modelo_path = modelo_path
        self.modelos = {}
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        # Crear directorio para modelos si no existe
        if not os.path.exists(modelo_path):
            os.makedirs(modelo_path)
        
        # Cargar datos
        self.datos_c19 = pd.read_csv('datos_CYP2C19.tsv', sep='\t')
        self.datos_d6 = pd.read_csv('datos_CYP2D6.tsv', sep='\t')
        
        # Resetear índices para evitar duplicados
        self.datos_c19 = self.datos_c19.reset_index(drop=True)
        self.datos_d6 = self.datos_d6.reset_index(drop=True)
        
        # Inicializar encoders para variables categóricas
        self._inicializar_encoders()
        
        # Preparar datos para entrenamiento
        self._preparar_datos()

    def _inicializar_encoders(self):
        """Inicializa los encoders para variables categóricas."""
        # Encoder para genotipos
        self.label_encoders['genotipo'] = LabelEncoder()
        todos_genotipos = pd.concat([
            self.datos_c19['genotipo_expandido'],
            self.datos_d6['genotipo_expandido']
        ]).fillna('Unknown').unique()
        self.label_encoders['genotipo'].fit(todos_genotipos)
        
        # Encoder para evaluaciones (solo las clases relevantes)
        self.label_encoders['evaluacion'] = LabelEncoder()
        self.label_encoders['evaluacion'].fit(['Positiva', 'Intermedia', 'Negativa'])
        
        # Encoder para categorías
        self.label_encoders['categoria'] = LabelEncoder()
        todas_categorias = pd.concat([
            self.datos_c19['phenotype_categories'],
            self.datos_d6['phenotype_categories']
        ]).fillna('Unknown').str.split(',').explode().fillna('Unknown').unique()
        self.label_encoders['categoria'].fit(todas_categorias)

    def _preparar_datos(self):
        """Prepara los datos para el entrenamiento de los modelos."""
        # Combinar datos de ambos genes
        self.datos_combinados = pd.concat([
            self.datos_c19.assign(gen='CYP2C19'),
            self.datos_d6.assign(gen='CYP2D6')
        ], ignore_index=True)  # Usar ignore_index para evitar duplicados
        
        # Crear características para el modelo
        self.X = self._crear_caracteristicas()
        # Reemplazar NaN en Evaluacion por 'Intermedia'
        y_labels = self.datos_combinados['Evaluacion'].fillna('Intermedia')
        self.y = self.label_encoders['evaluacion'].transform(y_labels)
        
        # Dividir datos en entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Escalar características
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def _crear_caracteristicas(self) -> pd.DataFrame:
        """Crea las características para el modelo."""
        # Crear DataFrame con índice limpio
        X = pd.DataFrame(index=range(len(self.datos_combinados)))
        
        # Codificar variables categóricas
        X['genotipo_cod'] = self.label_encoders['genotipo'].transform(self.datos_combinados['genotipo_expandido'])
        
        # Manejar valores NaN en categorías
        categorias = self.datos_combinados['phenotype_categories'].fillna('Unknown')
        X['categoria_cod'] = self.label_encoders['categoria'].transform(
            categorias.str.split(',').str[0].fillna('Unknown')
        )
        
        X['gen_cod'] = (self.datos_combinados['gen'].values == 'CYP2C19').astype(int)
        
        # Mantener las columnas de p-value que el modelo espera
        X['p_value'] = 0.5  # Valor por defecto
        X['log_p_value'] = -np.log10(0.5 + 1e-10)  # Transformación logarítmica del valor por defecto
        X['p_value_cat'] = 4  # Categoría por defecto (p-value > 0.05)
        
        return X

    def entrenar_modelos(self):
        """Entrena diferentes modelos de aprendizaje automático."""
        # Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(self.X_train_scaled, self.y_train)
        self.modelos['random_forest'] = rf_model
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(random_state=42)
        xgb_model.fit(self.X_train_scaled, self.y_train)
        self.modelos['xgboost'] = xgb_model
        
        # Gradient Boosting
        gb_model = GradientBoostingClassifier(random_state=42)
        gb_model.fit(self.X_train_scaled, self.y_train)
        self.modelos['gradient_boosting'] = gb_model
        
        # Evaluar modelos
        self._evaluar_modelos()
        
        # Guardar modelos
        self._guardar_modelos()

    def _evaluar_modelos(self):
        """Evalúa el rendimiento de los modelos entrenados."""
        for nombre, modelo in self.modelos.items():
            y_pred = modelo.predict(self.X_test_scaled)
            accuracy = accuracy_score(self.y_test, y_pred)
            print(f"\nEvaluación del modelo {nombre}:")
            print(f"Precisión: {accuracy:.3f}")
            print("\nReporte de clasificación:")
            print(classification_report(
                self.y_test, 
                y_pred,
                labels=[0,1,2,3],
                target_names=self.label_encoders['evaluacion'].classes_
            ))

    def _guardar_modelos(self):
        """Guarda los modelos entrenados."""
        for nombre, modelo in self.modelos.items():
            joblib.dump(modelo, f"{self.modelo_path}/{nombre}.joblib")
        joblib.dump(self.scaler, f"{self.modelo_path}/scaler.joblib")
        joblib.dump(self.label_encoders, f"{self.modelo_path}/label_encoders.joblib")

    def cargar_modelos(self):
        """Carga los modelos guardados."""
        for nombre in ['random_forest', 'xgboost', 'gradient_boosting']:
            self.modelos[nombre] = joblib.load(f"{self.modelo_path}/{nombre}.joblib")
        self.scaler = joblib.load(f"{self.modelo_path}/scaler.joblib")
        self.label_encoders = joblib.load(f"{self.modelo_path}/label_encoders.joblib")

    def _crear_caracteristicas_por_gen(self, datos: pd.DataFrame, gen: str) -> pd.DataFrame:
        """Crea características para un gen específico."""
        # Crear DataFrame con índice limpio
        X = pd.DataFrame(index=range(len(datos)))
        
        X['genotipo_cod'] = self.label_encoders['genotipo'].transform(
            datos['genotipo_expandido'].fillna('Unknown')
        )
        
        # Manejar valores NaN en categorías
        categorias = datos['phenotype_categories'].fillna('Unknown')
        X['categoria_cod'] = self.label_encoders['categoria'].transform(
            categorias.str.split(',').str[0].fillna('Unknown')
        )
        
        X['gen_cod'] = int(gen == 'CYP2C19')
        
        # Mantener las columnas de p-value que el modelo espera
        X['p_value'] = 0.5  # Valor por defecto
        X['log_p_value'] = -np.log10(0.5 + 1e-10)  # Transformación logarítmica del valor por defecto
        X['p_value_cat'] = 4  # Categoría por defecto (p-value > 0.05)
        
        return X

    def _obtener_p_value(self, farmaco: str, gen: str) -> float:
        """
        Obtiene el p-value para un fármaco y gen específicos.
        
        Args:
            farmaco: Nombre del fármaco
            gen: Gen a analizar ('CYP2C19' o 'CYP2D6')
            
        Returns:
            float: Valor del p-value o 0.5 si no se encuentra
        """
        try:
            # Seleccionar el conjunto de datos correcto
            datos = self.datos_c19 if gen == 'CYP2C19' else self.datos_d6
            
            # Función para buscar el fármaco en la lista de fármacos
            def contiene_farmaco(drugs):
                if pd.isna(drugs):
                    return False
                lista = [d.strip().lower() for d in str(drugs).split(';')]
                return farmaco.strip().lower() in lista
            
            # Filtrar datos para el fármaco específico usando la función personalizada
            datos_filtrados = datos[datos['drugs'].apply(contiene_farmaco)]
            
            if datos_filtrados.empty or 'p-value' not in datos_filtrados.columns:
                return 0.5
                
            # Obtener el primer p-value válido
            p_values = datos_filtrados['p-value'].dropna()
            if p_values.empty:
                return 0.5
                
            p_value_str = str(p_values.iloc[0]).strip()
            
            # Procesar diferentes formatos de p-value
            if p_value_str.startswith('='):
                try:
                    valor = float(p_value_str.split('=')[-1].strip())
                    return valor if valor > 0 else 0.5
                except:
                    return 0.5
                    
            if p_value_str.startswith('<'):
                try:
                    valor = float(p_value_str.split('<')[-1].strip())
                    return valor * 0.5 if valor > 0 else 0.5
                except:
                    return 0.5
                    
            try:
                valor = float(p_value_str)
                return valor if valor > 0 else 0.5
            except:
                return 0.5
                
        except Exception as e:
            print(f"Error al obtener p-value para {farmaco} en {gen}: {str(e)}")
            return 0.5

    def _obtener_predicciones_modelos(self, X_pred_scaled: np.ndarray, farmaco: str, gen: str) -> Dict[str, Dict]:
        """Obtiene predicciones de todos los modelos para un conjunto de características."""
        predicciones = {}
        
        # Obtener p-value usando la nueva función
        p_value = self._obtener_p_value(farmaco, gen)
        
        # Actualizar las características de p-value en X_pred_scaled
        # Nota: Esto es necesario porque el modelo espera estas características
        # pero no las usamos para la predicción final
        X_pred = pd.DataFrame(X_pred_scaled, columns=self.X.columns)
        X_pred['p_value'] = p_value
        X_pred['log_p_value'] = -np.log10(p_value + 1e-10)
        X_pred['p_value_cat'] = pd.cut(
            [p_value],
            bins=[0, 0.001, 0.01, 0.05, 0.1, 1],
            labels=[0, 1, 2, 3, 4]
        ).fillna(4).astype(int)[0]
        X_pred_scaled = self.scaler.transform(X_pred)
            
        for nombre, modelo in self.modelos.items():
            probas = modelo.predict_proba(X_pred_scaled)
            eval_pred = self.label_encoders['evaluacion'].inverse_transform(np.argmax(probas, axis=1))
            confianza = np.max(probas, axis=1)
            
            predicciones[nombre] = {
                'evaluacion': eval_pred[0],
                'confianza': confianza[0],
                'probabilidades': dict(zip(self.label_encoders['evaluacion'].classes_, probas[0])),
                'p_value': float(p_value)
            }
            
        return predicciones

    def predecir_evaluacion(self, genotipo_c19: str, genotipo_d6: str, farmaco: str) -> Dict[str, Dict]:
        """
        Predice la evaluación de un fármaco para los genotipos dados.
        
        Args:
            genotipo_c19: Genotipo de CYP2C19
            genotipo_d6: Genotipo de CYP2D6
            farmaco: Nombre del fármaco
            
        Returns:
            Diccionario con las predicciones de cada modelo para cada gen
        """
        predicciones = {
            'CYP2C19': {},
            'CYP2D6': {}
        }
        
        # Verificar en qué genes está presente el fármaco
        presente_c19 = farmaco in '; '.join(self.datos_c19['drugs'].dropna())
        presente_d6 = farmaco in '; '.join(self.datos_d6['drugs'].dropna())
        
        # Preparar y predecir para CYP2C19 si está presente y el genotipo es válido
        if presente_c19 and genotipo_c19 in self.label_encoders['genotipo'].classes_:
            X_pred_c19 = self._preparar_caracteristicas_prediccion_gen(genotipo_c19, farmaco, 'CYP2C19')
            if X_pred_c19 is not None and not X_pred_c19.empty:
                X_pred_c19 = X_pred_c19.fillna(self.X.mean())
                X_pred_c19_scaled = self.scaler.transform(X_pred_c19)
                predicciones['CYP2C19'] = self._obtener_predicciones_modelos(X_pred_c19_scaled, farmaco, 'CYP2C19')
        
        # Preparar y predecir para CYP2D6 si está presente y el genotipo es válido
        if presente_d6 and genotipo_d6 in self.label_encoders['genotipo'].classes_:
            X_pred_d6 = self._preparar_caracteristicas_prediccion_gen(genotipo_d6, farmaco, 'CYP2D6')
            if X_pred_d6 is not None and not X_pred_d6.empty:
                X_pred_d6 = X_pred_d6.fillna(self.X.mean())
                X_pred_d6_scaled = self.scaler.transform(X_pred_d6)
                predicciones['CYP2D6'] = self._obtener_predicciones_modelos(X_pred_d6_scaled, farmaco, 'CYP2D6')
        
        return predicciones

    def _preparar_caracteristicas_prediccion_gen(
        self, 
        genotipo: str, 
        farmaco: str,
        gen: str
    ) -> pd.DataFrame:
        """Prepara las características para la predicción de un gen específico."""
        try:
            # Normalizar farmaco y genotipo
            farmaco_norm = farmaco.strip().lower()
            genotipo_norm = genotipo.strip().lower()
            
            # Crear una copia explícita de los datos del gen
            datos_gen = self.datos_combinados[
                self.datos_combinados['gen'] == gen
            ].copy()
            
            # Normalizar columna drugs y buscar coincidencia exacta en la lista de fármacos
            def contiene_farmaco(drugs):
                if pd.isna(drugs):
                    return False
                lista = [d.strip().lower() for d in str(drugs).split(';')]
                return farmaco_norm in lista
            
            # Filtrar por fármaco y crear una copia explícita
            mascara_farmaco = datos_gen['drugs'].apply(contiene_farmaco)
            datos_farmaco = datos_gen[mascara_farmaco].copy()
            
            if datos_farmaco.empty:
                return None
            
            # Crear una nueva columna usando .loc para evitar el warning
            datos_farmaco.loc[:, 'genotipo_expandido_norm'] = (
                datos_farmaco['genotipo_expandido']
                .astype(str)
                .str.strip()
                .str.lower()
            )
            
            # Buscar coincidencia exacta y crear una copia explícita
            mascara_genotipo = datos_farmaco['genotipo_expandido_norm'] == genotipo_norm
            datos_genotipo = datos_farmaco[mascara_genotipo].copy()
            
            if datos_genotipo.empty:
                # Si no hay coincidencia exacta, buscar la entrada más relevante
                # Priorizar entradas con p-value y evaluaciones más recientes
                datos_genotipo = (
                    datos_farmaco
                    .sort_values(by=['p-value', 'Evaluacion'], ascending=[True, False])
                    .head(1)
                    .copy()
                )
            
            if datos_genotipo.empty:
                return None
            
            # Crear características para todas las entradas encontradas
            X_list = []
            for _, row in datos_genotipo.iterrows():
                X_temp = self._crear_caracteristicas_por_gen(
                    pd.DataFrame([row]).copy(), 
                    gen
                )
                if X_temp is not None and not X_temp.empty:
                    X_list.append(X_temp)
            
            if not X_list:
                return None
                
            # Combinar todas las características encontradas
            X = pd.concat(X_list, ignore_index=True)
            
            # Si hay múltiples entradas, tomar la que tenga el p-value más significativo
            if len(X) > 1:
                X = X.sort_values('p_value').head(1).copy()
            
            return X
            
        except Exception as e:
            print(f"Error en _preparar_caracteristicas_prediccion_gen: {str(e)}")
            return None

    def recomendar_farmacos(self, genotipo_c19: str, genotipo_d6: str, top_n: int = 5) -> List[List[Dict]]:
        """
        Genera recomendaciones de fármacos usando los modelos de IA, retornando los mejores y peores.
        
        Args:
            genotipo_c19: Genotipo de CYP2C19
            genotipo_d6: Genotipo de CYP2D6
            top_n (int, opcional): Número de recomendaciones a retornar (por defecto 5).
            
        Returns:
            Lista con dos sublistas: [mejores, peores] (cada una con top_n recomendaciones).
        """
        todos_farmacos = pd.concat([
            self.datos_c19['drugs'],
            self.datos_d6['drugs']
        ]).str.split('; ').explode().dropna().unique()
        
        recomendaciones = []
        for farmaco in todos_farmacos:
            if not isinstance(farmaco, str) or not farmaco.strip():
                continue
                
            # Verificar en qué genes está presente el fármaco y su frecuencia
            presente_c19 = farmaco in '; '.join(self.datos_c19['drugs'].dropna())
            presente_d6 = farmaco in '; '.join(self.datos_d6['drugs'].dropna())
            
            # Obtener predicciones para cada gen
            predicciones = self.predecir_evaluacion(genotipo_c19, genotipo_d6, farmaco)
            
            # Calcular puntuación para cada gen con sus confianzas
            puntuacion_c19, confianza_c19 = self._calcular_puntuacion_gen(predicciones.get('CYP2C19', {}))
            puntuacion_d6, confianza_d6 = self._calcular_puntuacion_gen(predicciones.get('CYP2D6', {}))
            
            # Calcular puntuación final ponderada considerando la confianza
            if presente_c19 and presente_d6:
                # Ponderar por confianza de cada gen
                peso_total = confianza_c19 + confianza_d6
                if peso_total > 0:
                    puntuacion = (puntuacion_c19 * confianza_c19 + puntuacion_d6 * confianza_d6) / peso_total
                else:
                    puntuacion = 0.0  # Si no hay confianza en ninguna predicción
            elif presente_c19:
                # Si solo está en CYP2C19, usar su puntuación con un factor de ajuste
                if confianza_c19 > 0.3:
                    # Aplicar un factor de ajuste de 0.8 para reflejar la falta de información del otro gen
                    puntuacion = puntuacion_c19 * 0.8
                else:
                    puntuacion = 0.0
            elif presente_d6:
                # Si solo está en CYP2D6, usar su puntuación con un factor de ajuste
                if confianza_d6 > 0.3:
                    # Aplicar un factor de ajuste de 0.8 para reflejar la falta de información del otro gen
                    puntuacion = puntuacion_d6 * 0.8
                else:
                    puntuacion = 0.0
            else:
                puntuacion = 0.0  # Si no está en ningún gen, no recomendamos
            
            # Solo incluir recomendaciones con puntuación válida
            if puntuacion > 0:
                recomendaciones.append({
                    'farmaco': farmaco,
                    'puntuacion': puntuacion,
                    'predicciones': predicciones,
                    'presente_en': {
                        'CYP2C19': presente_c19,
                        'CYP2D6': presente_d6
                    },
                    'confianzas': {
                        'CYP2C19': confianza_c19,
                        'CYP2D6': confianza_d6
                    },
                    'factor_ajuste': 0.8 if (presente_c19 or presente_d6) and not (presente_c19 and presente_d6) else 1.0
                })
        
        # Ordenar recomendaciones por puntuación
        recomendaciones_ordenadas = sorted(recomendaciones, key=lambda x: x['puntuacion'], reverse=True)
        
        # Filtrar recomendaciones con puntuación muy baja
        recomendaciones_filtradas = [r for r in recomendaciones_ordenadas if r['puntuacion'] > 0.3]
        
        # Si no hay suficientes recomendaciones válidas, usar las originales
        if len(recomendaciones_filtradas) < top_n:
            recomendaciones_filtradas = recomendaciones_ordenadas
        
        mejores = recomendaciones_filtradas[:top_n]
        peores = recomendaciones_ordenadas[-top_n:]
        return [mejores, peores]

    def _calcular_puntuacion_gen(self, predicciones_gen: Dict[str, Dict]) -> tuple[float, float]:
        """
        Calcula la puntuación y confianza promedio para un gen específico basado en sus predicciones.
        Incorpora el p-value como factor de ajuste en la puntuación.
        
        Returns:
            tuple: (puntuacion, confianza_promedio)
        """
        if not predicciones_gen or not any(predicciones_gen.values()):
            return 0.0, 0.0  # Sin predicciones válidas
            
        puntuacion = 0.0
        suma_confianza = 0.0
        num_predicciones_validas = 0
        
        for modelo, pred in predicciones_gen.items():
            if not pred or 'evaluacion' not in pred or 'confianza' not in pred:
                continue
                
            # Solo considerar predicciones con confianza suficiente
            if pred['confianza'] < 0.3:
                continue
                
            # Obtener el p-value y calcular el factor de ajuste
            p_value = pred.get('p_value', 0.5)  # Valor por defecto si no hay p-value
            # Transformar p-value a un factor entre 0.5 y 1.5
            # p-value < 0.001 -> factor = 1.5 (máxima confianza)
            # p-value < 0.01  -> factor = 1.3 (confianza alta)
            # p-value < 0.05  -> factor = 1.2 (confianza moderada alta)
            # p-value < 0.1   -> factor = 1.1 (confianza moderada)
            # p-value >= 0.1  -> factor = 0.5 (mínima confianza)
            if p_value < 0.001:
                factor_p_value = 1.5
            elif p_value < 0.01:
                factor_p_value = 1.3
            elif p_value < 0.05:
                factor_p_value = 1.2
            elif p_value < 0.1:
                factor_p_value = 1.1
            else:
                factor_p_value = 0.5
            
            peso = pred['confianza'] * factor_p_value
            suma_confianza += peso
            num_predicciones_validas += 1
            
            if pred['evaluacion'] == 'Positiva':
                puntuacion += 3.0 * peso
            elif pred['evaluacion'] == 'Intermedia':
                puntuacion += 2.0 * peso
            else:  # Negativa
                puntuacion += 1.0 * peso
        
        if num_predicciones_validas > 0 and suma_confianza > 0:
            confianza_promedio = suma_confianza / num_predicciones_validas
            return puntuacion / suma_confianza, confianza_promedio
            
        return 0.0, 0.0  # Sin predicciones válidas o confianza suficiente 