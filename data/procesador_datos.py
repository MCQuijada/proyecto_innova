import pandas as pd
import numpy as np
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
        # import re # re no es necesario aquí
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
            
        return df_exp 