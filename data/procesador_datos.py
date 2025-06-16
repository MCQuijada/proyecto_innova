import pandas as pd
import re

class ProcesadorDatos:
    def __init__(self):
        self.datos_original = None    # Datos de variantes originales
        self.datos_procesados = None  # Datos después del procesamiento
        
    def cargar_datos(self, ruta_archivo):
        """
        Carga los datos desde un archivo TSV
        Args:
            ruta_archivo (str): Ruta al archivo de datos TSV
        Returns:
            DataFrame: Datos cargados
        """
        try:
            if not ruta_archivo.endswith('.tsv'):
                print("Advertencia: El archivo no tiene extensión .tsv")
            
            self.datos_original = pd.read_csv(ruta_archivo, sep='\t', encoding='utf-8')
            
            print("\nDatos cargados exitosamente:")
            print(f"Número de filas: {len(self.datos_original)}")
            print(f"Número de columnas: {len(self.datos_original.columns)}")
            print("\nColumnas disponibles:")
            print(list(self.datos_original.columns))
            
            return self.datos_original
            
        except Exception as e:
            print(f"Error al cargar los datos: {str(e)}")
            return None

    def crear_columna_genotipo(self):
        """
        Crea la columna Genotipo como una lista de todos los genotipos posibles detectados en la columna Variant,
        usando las combinaciones de variantes y equivalencias definidas (por ejemplo, CYP2D6*1, CYP2D6*4, rs3892097, etc.).
        Si no se detecta ninguna combinación, retorna [].
        Returns:
            DataFrame: Datos con la nueva columna Genotipo (siempre lista)
        """
        if self.datos_original is None:
            print("Error: No hay datos cargados")
            return None
        try:
            self.datos_procesados = self.datos_original.copy()
            if 'Variant' not in self.datos_procesados.columns:
                print("Error: No se encontró la columna 'Variant' en los datos")
                return None

            # Equivalencias de variantes para CYP2D6 y CYP2C19
            equivalencias = {
                # CYP2D6
                'CYP2D6*4': ['CYP2D6*4', 'rs3892097'],
                'CYP2D6*1': ['CYP2D6*1'],
                # CYP2C19
                'CYP2C19*1': ['CYP2C19*1'],
                'CYP2C19*2': ['CYP2C19*2', 'rs4244285'],
                'CYP2C19*17': ['CYP2C19*17', 'rs12248560'],
            }

            # Todas las combinaciones de variantes a genotipo (CYP2D6 y CYP2C19)
            todas_las_combinaciones = {
                frozenset(['CYP2D6*4']): 'A/A',
                frozenset(['CYP2D6*1']): 'G/G',
                frozenset(['CYP2D6*1', 'CYP2D6*4']): 'G/A',
                frozenset(['CYP2C19*1']): 'G/G + C/C',
                frozenset(['CYP2C19*1', 'CYP2C19*2']): 'A/G + C/C',
                frozenset(['CYP2C19*2']): 'A/A + C/C',
                frozenset(['CYP2C19*1', 'CYP2C19*17']): 'G/G + C/T',
                frozenset(['CYP2C19*17']): 'G/G + T/T',
                frozenset(['CYP2C19*2', 'CYP2C19*17']): 'A/G + C/T',
            }

            def normalizar_variant(variant):
                """
                Normaliza las variantes para que sean consistentes con las equivalencias definidas.
                Args:
                    variant (str): Variante a normalizar
                Returns:
                    set: Conjunto de variantes normalizadas
                """
                if pd.isna(variant):
                    return set()
                # Dividir por ; o paréntesis, y limpiar espacios
                partes = [v.strip() for v in re.split(r';|\(|\)', str(variant)) if v.strip()]
                resultado = set()
                for parte in partes:
                    for key, vals in equivalencias.items():
                        for val in vals:
                            # Buscar val como palabra completa o si está entre paréntesis
                            if re.search(rf'\b{re.escape(val)}\b', parte) or f'({val})' in parte:
                                resultado.add(key)
                return resultado

            def detectar_genotipos(row):
                """
                Detecta los genotipos posibles para una fila dada.
                Args:
                    row (pd.Series): Fila del DataFrame
                Returns:
                    list: Lista de genotipos detectados
                """
                variantes = normalizar_variant(row['Variant'])
                genotipos_detectados = []
                # Detección general para todas las combinaciones
                for comb, genotipo in todas_las_combinaciones.items():
                    if variantes.issuperset(comb):
                        genotipos_detectados.append(genotipo)
                        
                return genotipos_detectados

            self.datos_procesados['genotipo'] = self.datos_procesados.apply(detectar_genotipos, axis=1)
            total_registros = len(self.datos_procesados)
            registros_con_genotipo = self.datos_procesados['genotipo'].apply(lambda x: len(x) > 0).sum()
            print(f"\nCreación de columna Genotipo (por combinaciones):")
            print(f"Total de registros: {total_registros}")
            print(f"Registros con al menos un genotipo detectado: {registros_con_genotipo}")
            print(f"Porcentaje de registros con genotipo: {(registros_con_genotipo/total_registros)*100:.2f}%")
            return self.datos_procesados
        except Exception as e:
            print(f"Error al crear la columna Genotipo: {str(e)}")
            return None

    def limpiar_datos(self):
        """
        Limpia los datos eliminando columnas no necesarias y estandarizando variables
        """
        if self.datos_procesados is None:
            print("Error: No hay datos procesados")
            return None

        try:
            # Columnas a eliminar
            columnas_a_eliminar = [
                'PharmGKB ID', 'Literature', 'Pediatric', 'More Details',
                '# of Cases', '# of Controls', 'Significance', 'Association',
                'Biogeographical Groups'
            ]
            columnas_existentes = [col for col in columnas_a_eliminar if col in self.datos_procesados.columns]
            
            if columnas_existentes:
                self.datos_procesados = self.datos_procesados.drop(columns=columnas_existentes)
                print(f"\nColumnas eliminadas: {columnas_existentes}")
            
            # Estandarizar nombres de columnas
            self.datos_procesados.columns = self.datos_procesados.columns.str.lower().str.replace(' ', '_')
            
            # Mostrar información de las columnas después de la limpieza
            print("\nColumnas después de la limpieza:")
            print(list(self.datos_procesados.columns))
            
            return self.datos_procesados

        except Exception as e:
            print(f"Error al limpiar los datos: {str(e)}")
            return None

    def verificar_datos(self):
        """
        Verifica la calidad de los datos procesados
        Returns:
            dict: Diccionario con estadísticas de los datos
        """
        if self.datos_procesados is None:
            print("Error: No hay datos procesados para verificar")
            return None

        try:
            estadisticas = {
                'valores_faltantes': self.datos_procesados.isnull().sum().to_dict(),
                'tipos_datos': self.datos_procesados.dtypes.to_dict(),
                'estadisticas_numericas': self.datos_procesados.describe().to_dict(),
                'valores_unicos': {col: self.datos_procesados[col].nunique() 
                                 for col in self.datos_procesados.columns}
            }
            
            print("\n=== ESTADÍSTICAS DE LOS DATOS ===")
            print("\nValores faltantes por columna:")
            print(estadisticas['valores_faltantes'])
            
            print("\nTipos de datos por columna:")
            print(estadisticas['tipos_datos'])
            
            print("\nNúmero de valores únicos por columna:")
            print(estadisticas['valores_unicos'])
            
            # Mostrar distribución de la columna Genotipo
            if 'genotipo' in self.datos_procesados.columns:
                print("\nDistribución de Genotipos:")
                print(self.datos_procesados['genotipo'].value_counts())
            
            return estadisticas

        except Exception as e:
            print(f"Error al verificar los datos: {str(e)}")
            return None

    def obtener_datos_procesados(self):
        """
        Retorna los datos procesados
        Returns:
            DataFrame: Datos procesados
        """
        if self.datos_procesados is None:
            print("Advertencia: Los datos no han sido procesados")
            return self.datos_original
        return self.datos_procesados 

    def guardar_datos_procesados(self, ruta_archivo):
        """
        Guarda los datos procesados en un archivo TSV
        Args:
            ruta_archivo (str): Ruta donde guardar los datos
        """
        if self.datos_procesados is None:
            print("Error: No hay datos procesados para guardar")
            return False
        try:
            self.datos_procesados.to_csv(ruta_archivo, sep='\t', index=False)
            print(f"\nDatos procesados guardados exitosamente en: {ruta_archivo}")
            return True
        except Exception as e:
            print(f"Error al guardar los datos: {str(e)}")
            return False 

    def expandir_genotipos(self):
        """
        Expande el DataFrame para que cada fila tenga solo un genotipo detectado.
        Si la lista de genotipos está vacía, se coloca 'No detectado' al expandir.
        Elimina la columna original 'genotipo'.
        Returns:
            DataFrame: DataFrame expandido por genotipo
        """
        if self.datos_procesados is None:
            print("Error: No hay datos procesados para expandir genotipos")
            return None
        import pandas as pd

        df = self.datos_procesados.copy()
        
        # Asegura que la columna 'genotipo' sea una lista (incluso vacía)
        df['genotipo'] = df['genotipo'].apply(lambda x: x if isinstance(x, list) else [])

        # Función para asignar genotipos o 'No detectado'
        def genotipos_o_no_detectado(genotipo_list):
            if genotipo_list:
                return genotipo_list
            return ['No detectado']

        # Aplica la función a la columna 'genotipo' para preparar la expansión
        df['genotipo_expandido'] = df['genotipo'].apply(genotipos_o_no_detectado)

        # Expandir el DataFrame
        df_exp = df.explode('genotipo_expandido').reset_index(drop=True)
        
        # Eliminar columna original 'genotipo'
        if 'genotipo' in df_exp.columns:
            df_exp = df_exp.drop(columns=['genotipo'])
            
        # Actualizar datos_procesados con el DataFrame expandido
        self.datos_procesados = df_exp
        
        return df_exp

    def crear_columna_evaluacion(self):
        """
        Crea la columna Evaluacion basada en las categorías y genotipos para CYP2D6 y CYP2C19.
        Returns:
            tuple: (DataFrame CYP2D6, DataFrame CYP2C19) con la columna Evaluacion agregada
        """
        if self.datos_procesados is None:
            print("Error: No hay datos procesados para crear evaluación")
            return None, None

        try:
            # Definición de prioridades de categorías (común para todos los genes)
            prioridad = ['Metabolism/PK', 'Efficacy', 'Dosage', 'Toxicity', 'Other']

            def normalize_genotype_key(genotype_str):
                if pd.isna(genotype_str):
                    return None
                genotype_str = str(genotype_str).strip()
                # Normalizar espacios alrededor de '+' a un solo espacio
                normalized = re.sub(r'\s*\+\s*', ' + ', genotype_str)
                return normalized

            def seleccionar_categoria(cat_str):
                if pd.isna(cat_str):
                    return None
                cat_str = str(cat_str).strip()
                normalized_cat_str = re.sub(r'\s*/\s*', '/', cat_str)
                normalized_cat_str = re.sub(r'\s*,\s*', ',', normalized_cat_str)
                
                for p_cat in prioridad:
                    if p_cat == normalized_cat_str:
                        return p_cat
                
                if ',' in normalized_cat_str:
                    parts = [p.strip() for p in normalized_cat_str.split(',') if p.strip()]
                    for p_cat_priority in prioridad:
                        if p_cat_priority in parts:
                            return p_cat_priority
                
                return None

            # --- Análisis para CYP2D6 ---
            datos_cyp2d6 = None
            if 'genes' in self.datos_procesados.columns:
                datos_cyp2d6 = self.datos_procesados[(self.datos_procesados['genes'].str.upper() == 'CYP2D6')].copy()
                eval_map_cyp2d6 = {
                    ('Efficacy', 'G/G'): 'Intermedia',
                    ('Efficacy', 'G/A'): 'Positiva',
                    ('Efficacy', 'A/A'): 'Negativa',
                    ('Dosage', 'G/G'): 'Intermedia',
                    ('Dosage', 'G/A'): 'Positiva',
                    ('Dosage', 'A/A'): 'Negativa',
                    ('Toxicity', 'G/G'): 'Negativa',
                    ('Toxicity', 'G/A'): 'Intermedia',
                    ('Toxicity', 'A/A'): 'Positiva',
                    ('Metabolism/PK', 'G/G'): 'Positiva',
                    ('Metabolism/PK', 'G/A'): 'Intermedia',
                    ('Metabolism/PK', 'A/A'): 'Negativa',
                }
                def asignar_evaluacion_cyp2d6(row):
                    cat = seleccionar_categoria(row['phenotype_categories'])
                    if cat == 'Other':
                        return 'Intermedia'
                    key = (cat, normalize_genotype_key(row['genotipo_expandido']))
                    return eval_map_cyp2d6.get(key, None)
                datos_cyp2d6['Evaluacion'] = datos_cyp2d6.apply(asignar_evaluacion_cyp2d6, axis=1)
                # Filtrar filas donde Evaluacion no es None
                datos_cyp2d6 = datos_cyp2d6[datos_cyp2d6['Evaluacion'].notna()]
                datos_cyp2d6.to_csv('datos_CYP2D6.tsv', sep='\t', index=False)
                print("\nArchivo generado: datos_CYP2D6.tsv (expandido por genotipo)")
                print("\nResumen de Evaluaciones para CYP2D6:")
                print(datos_cyp2d6['Evaluacion'].value_counts(dropna=False))
            else:
                print("Advertencia: No se encontró la columna 'genes' en los datos procesados para CYP2D6.")

            # --- Análisis para CYP2C19 ---
            datos_cyp2c19 = None
            if 'genes' in self.datos_procesados.columns:
                datos_cyp2c19 = self.datos_procesados[(self.datos_procesados['genes'].str.upper() == 'CYP2C19')].copy()
                eval_map_cyp2c19 = {
                    # Efficacy y Dosage
                    ('Efficacy', 'G/G + C/C'): 'Intermedia',
                    ('Efficacy', 'A/G + C/C'): 'Positiva',
                    ('Efficacy', 'A/G + C/T'): 'Positiva',
                    ('Efficacy', 'A/A + C/C'): 'Intermedia',
                    ('Efficacy', 'G/G + T/T'): 'Negativa',
                    ('Efficacy', 'G/G + C/T'): 'Negativa',
                    
                    ('Dosage', 'G/G + C/C'): 'Intermedia',
                    ('Dosage', 'A/G + C/C'): 'Positiva',
                    ('Dosage', 'A/G + C/T'): 'Positiva',
                    ('Dosage', 'A/A + C/C'): 'Intermedia',
                    ('Dosage', 'G/G + T/T'): 'Negativa',
                    ('Dosage', 'G/G + C/T'): 'Negativa',

                    # Toxicity
                    ('Toxicity', 'G/G + C/C'): 'Negativa',
                    ('Toxicity', 'A/G + C/C'): 'Negativa',
                    ('Toxicity', 'A/G + C/T'): 'Negativa',
                    ('Toxicity', 'A/A + C/C'): 'Intermedia',
                    ('Toxicity', 'G/G + T/T'): 'Intermedia',
                    ('Toxicity', 'G/G + C/T'): 'Intermedia',

                    # Metabolism/Pk
                    ('Metabolism/PK', 'G/G + C/C'): 'Positiva',
                    ('Metabolism/PK', 'A/G + C/C'): 'Intermedia',
                    ('Metabolism/PK', 'A/G + C/T'): 'Intermedia',
                    ('Metabolism/PK', 'A/A + C/C'): 'Negativa',
                    ('Metabolism/PK', 'G/G + T/T'): 'Positiva',
                    ('Metabolism/PK', 'G/G + C/T'): 'Positiva',
                }
                def asignar_evaluacion_cyp2c19(row):
                    cat = seleccionar_categoria(row['phenotype_categories'])
                    if cat == 'Other':
                        return 'Intermedia'
                    key = (cat, normalize_genotype_key(row['genotipo_expandido']))
                    return eval_map_cyp2c19.get(key, None)
                datos_cyp2c19['Evaluacion'] = datos_cyp2c19.apply(asignar_evaluacion_cyp2c19, axis=1)
                # Filtrar filas donde Evaluacion no es None
                datos_cyp2c19 = datos_cyp2c19[datos_cyp2c19['Evaluacion'].notna()]
                datos_cyp2c19.to_csv('datos_CYP2C19.tsv', sep='\t', index=False)
                print("\nArchivo generado: datos_CYP2C19.tsv (expandido por genotipo)")
                print("\nResumen de Evaluaciones para CYP2C19:")
                print(datos_cyp2c19['Evaluacion'].value_counts(dropna=False))
            else:
                print("Advertencia: No se encontró la columna 'genes' en los datos procesados para CYP2C19.")

            return datos_cyp2d6, datos_cyp2c19

        except Exception as e:
            print(f"Error al crear la columna Evaluacion: {str(e)}")
            return None, None 