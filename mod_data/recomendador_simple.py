import pandas as pd
from typing import Dict, List, Tuple

class RecomendadorSimple:
    def __init__(self):
        """
        Inicializa el recomendador simple basado en reglas.
        """
        # Cargar datos
        self.datos_c19 = pd.read_csv('datos_CYP2C19.tsv', sep='\t')
        self.datos_d6 = pd.read_csv('datos_CYP2D6.tsv', sep='\t')
        
        # Resetear índices para evitar duplicados
        self.datos_c19 = self.datos_c19.reset_index(drop=True)
        self.datos_d6 = self.datos_d6.reset_index(drop=True)
        
        # Definir puntuaciones base según evaluación
        self.PUNTUACIONES_BASE = {
            'Positiva': 1.0,
            'Intermedia': 0.5,
            'Negativa': 0.0
        }
        
        # Definir factores de p-value
        self.FACTORES_P_VALUE = {
            (0, 0.001): 1.5,    # p < 0.001: Máxima significancia
            (0.001, 0.01): 1.3, # p < 0.01: Alta significancia
            (0.01, 0.05): 1.2,  # p < 0.05: Moderada alta significancia
            (0.05, 0.1): 1.1,   # p < 0.1: Moderada significancia
            (0.1, float('inf')): 0.5  # p >= 0.1: Baja significancia
        }
        
        # Definir factores de ajuste por genes
        self.FACTOR_UN_GEN = 0.8  # Factor cuando el fármaco está en un solo gen
        self.FACTOR_DOS_GENES = 1.0  # Factor cuando el fármaco está en ambos genes
        
        # Puntuación máxima para cálculo de porcentaje
        self.PUNTUACION_MAXIMA = 1.5

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
            
            # Filtrar datos para el fármaco específico
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

    def _obtener_factor_p_value(self, p_value: float) -> float:
        """
        Obtiene el factor de ajuste basado en el p-value.
        
        Args:
            p_value: Valor del p-value
            
        Returns:
            float: Factor de ajuste correspondiente
        """
        for (min_val, max_val), factor in self.FACTORES_P_VALUE.items():
            if min_val <= p_value < max_val:
                return factor
        return 0.5  # Valor por defecto

    def _obtener_evaluacion(self, farmaco: str, gen: str, genotipo: str) -> Tuple[str, float]:
        """
        Obtiene la mejor evaluación y confianza para un fármaco, gen y genotipo específicos.
        Si hay varias filas para la misma combinación, se queda con la mejor evaluación (Positiva > Intermedia > Negativa)
        para cada categoría (Metabolism/PK, Efficacy, Dosage, Toxicity, Other).
        """
        try:
            datos = self.datos_c19 if gen == 'CYP2C19' else self.datos_d6
            mascara_farmaco = datos['drugs'].str.contains(farmaco, case=False, na=False)
            mascara_genotipo = datos['genotipo_expandido'] == genotipo
            datos_filtrados = datos[mascara_farmaco & mascara_genotipo]
            if datos_filtrados.empty:
                return None, 0.0
            # Definir prioridad de evaluaciones
            prioridad_eval = {'Positiva': 2, 'Intermedia': 1, 'Negativa': 0}
            # Si hay columna de categoría, agrupar por ella y quedarse con la mejor evaluación
            if 'phenotype_categories' in datos_filtrados.columns:
                mejores = []
                for cat, grupo in datos_filtrados.groupby('phenotype_categories'):
                    evaluaciones = grupo['Evaluacion'].dropna()
                    if not evaluaciones.empty:
                        # Elegir la mejor evaluación según prioridad
                        mejor_eval = sorted(evaluaciones, key=lambda x: prioridad_eval.get(x, -1), reverse=True)[0]
                        mejores.append(mejor_eval)
                if not mejores:
                    return None, 0.0
                # De todas las mejores por categoría, elegir la mejor global
                mejor_global = sorted(mejores, key=lambda x: prioridad_eval.get(x, -1), reverse=True)[0]
                # Calcular confianza como la proporción de esa evaluación entre todas las filas
                total = len(datos_filtrados)
                confianza = (datos_filtrados['Evaluacion'] == mejor_global).sum() / total
                return mejor_global, confianza
            # Si no hay columna de categoría, usar la mejor evaluación global
            evaluaciones = datos_filtrados['Evaluacion'].dropna()
            if evaluaciones.empty:
                return None, 0.0
            mejor_eval = sorted(evaluaciones, key=lambda x: prioridad_eval.get(x, -1), reverse=True)[0]
            confianza = (evaluaciones == mejor_eval).sum() / len(evaluaciones)
            return mejor_eval, confianza
        except Exception as e:
            print(f"Error al obtener evaluación para {farmaco} en {gen}: {str(e)}")
            return None, 0.0

    def _calcular_puntuacion_gen(self, farmaco: str, gen: str, genotipo: str) -> Tuple[float, float, Dict]:
        """
        Calcula la puntuación para un gen específico.
        
        Args:
            farmaco: Nombre del fármaco
            gen: Gen a analizar ('CYP2C19' o 'CYP2D6')
            genotipo: Genotipo a evaluar
            
        Returns:
            tuple: (puntuacion, confianza, detalles)
        """
        # Obtener evaluación y confianza
        evaluacion, confianza = self._obtener_evaluacion(farmaco, gen, genotipo)
        if evaluacion is None:
            return 0.0, 0.0, {}
        
        # Obtener p-value y su factor
        p_value = self._obtener_p_value(farmaco, gen)
        factor_p_value = self._obtener_factor_p_value(p_value)
        
        # Calcular puntuación base
        puntuacion_base = self.PUNTUACIONES_BASE.get(evaluacion, 0.0)
        
        # Aplicar factor de p-value
        puntuacion_final = puntuacion_base * factor_p_value
        
        # Preparar detalles
        detalles = {
            'evaluacion': evaluacion,
            'confianza': confianza,
            'p_value': p_value,
            'factor_p_value': factor_p_value,
            'puntuacion_base': puntuacion_base
        }
        
        return puntuacion_final, confianza, detalles

    def recomendar_farmacos(self, genotipo_c19: str, genotipo_d6: str, top_n: int = 10000) -> List[List[Dict]]:
        """
        Genera recomendaciones de fármacos basadas en los genotipos proporcionados.
        
        Args:
            genotipo_c19: Genotipo de CYP2C19
            genotipo_d6: Genotipo de CYP2D6
        Returns:
            Lista con dos sublistas: [mejores, peores] (todas las recomendaciones ordenadas)
        """
        # Obtener lista única de fármacos
        todos_farmacos = pd.concat([
            self.datos_c19['drugs'],
            self.datos_d6['drugs']
        ]).str.split('; ').explode().dropna().unique()
        recomendaciones = []
        for farmaco in todos_farmacos:
            if not isinstance(farmaco, str) or not farmaco.strip():
                continue
            presente_c19 = farmaco in '; '.join(self.datos_c19['drugs'].dropna())
            presente_d6 = farmaco in '; '.join(self.datos_d6['drugs'].dropna())
            puntuacion_c19, confianza_c19, detalles_c19 = self._calcular_puntuacion_gen(
                farmaco, 'CYP2C19', genotipo_c19
            ) if presente_c19 else (0.0, 0.0, {})
            puntuacion_d6, confianza_d6, detalles_d6 = self._calcular_puntuacion_gen(
                farmaco, 'CYP2D6', genotipo_d6
            ) if presente_d6 else (0.0, 0.0, {})
            if presente_c19 and presente_d6:
                peso_total = confianza_c19 + confianza_d6
                if peso_total > 0:
                    puntuacion = (puntuacion_c19 * confianza_c19 + puntuacion_d6 * confianza_d6) / peso_total
                    factor_ajuste = self.FACTOR_DOS_GENES
                else:
                    puntuacion = 0.0
                    factor_ajuste = 0.0
            elif presente_c19:
                puntuacion = puntuacion_c19 * self.FACTOR_UN_GEN
                factor_ajuste = self.FACTOR_UN_GEN
            elif presente_d6:
                puntuacion = puntuacion_d6 * self.FACTOR_UN_GEN
                factor_ajuste = self.FACTOR_UN_GEN
            else:
                puntuacion = 0.0
                factor_ajuste = 0.0
            porcentaje_exito = (puntuacion / self.PUNTUACION_MAXIMA) * 100 if puntuacion > 0 else 0.0
            if puntuacion > 0:
                recomendaciones.append({
                    'farmaco': farmaco,
                    'puntuacion': puntuacion,
                    'porcentaje_exito': porcentaje_exito,
                    'predicciones': {
                        'CYP2C19': detalles_c19 if presente_c19 else {},
                        'CYP2D6': detalles_d6 if presente_d6 else {}
                    },
                    'presente_en': {
                        'CYP2C19': presente_c19,
                        'CYP2D6': presente_d6
                    },
                    'confianzas': {
                        'CYP2C19': confianza_c19,
                        'CYP2D6': confianza_d6
                    },
                    'factor_ajuste': factor_ajuste
                })
        recomendaciones_ordenadas = sorted(recomendaciones, key=lambda x: x['puntuacion'], reverse=True)
        recomendaciones_filtradas = [r for r in recomendaciones_ordenadas if r['puntuacion'] > 0.3]
        if len(recomendaciones_filtradas) < 1:
            recomendaciones_filtradas = recomendaciones_ordenadas
        mejores = recomendaciones_filtradas
        peores = list(reversed(recomendaciones_filtradas))
        return [mejores, peores] 