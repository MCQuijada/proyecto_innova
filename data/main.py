from procesador_datos import ProcesadorDatos
from analizador_ia import AnalizadorIA
import pandas as pd
import re

def main():
    """
    Función principal que ejecuta el flujo completo del programa
    """
    print("=== INICIO DEL PROCESAMIENTO DE DATOS Y ANÁLISIS DE IA ===")
    
    # Crear instancias de las clases
    procesador = ProcesadorDatos()
    analizador = AnalizadorIA()
    
    # Cargar datos de variantes
    print("\n1. Cargando datos de variantes...")
    datos_variante = procesador.cargar_datos('datos_variante.tsv')
    
    if datos_variante is not None:
        # Crear columna Genotipo
        print("\n2. Creando columna Genotipo...")
        datos_con_genotipo = procesador.crear_columna_genotipo()
        
        if datos_con_genotipo is not None:
            # Limpiar y procesar los datos
            print("\n3. Limpiando y procesando datos...")
            datos_procesados = procesador.limpiar_datos()
            
            if datos_procesados is not None:
                # Expandir genotipos
                print("\n4. Expandiendo genotipos...")
                datos_expandidos = procesador.expandir_genotipos()
                
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
                if 'genes' in datos_expandidos.columns:
                    datos_cyp2d6 = datos_expandidos[(datos_expandidos['genes'].str.upper() == 'CYP2D6')].copy()
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
                    datos_cyp2d6.to_csv('datos_CYP2D6.tsv', sep='\t', index=False)
                    print("\nArchivo generado: datos_CYP2D6.tsv (expandido por genotipo)")
                    print("\nResumen de Evaluaciones para CYP2D6:")
                    print(datos_cyp2d6['Evaluacion'].value_counts(dropna=False))
                else:
                    print("Advertencia: No se encontró la columna 'genes' en los datos procesados para CYP2D6.")

                # --- Análisis para CYP2C19 ---
                if 'genes' in datos_expandidos.columns:
                    datos_cyp2c19 = datos_expandidos[(datos_expandidos['genes'].str.upper() == 'CYP2C19')].copy()
                    eval_map_cyp2c19 = {
                        # Efficacy y Dosage (mismas reglas de la nueva imagen)
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

                        # Toxicity (nueva imagen)
                        ('Toxicity', 'G/G + C/C'): 'Negativa',
                        ('Toxicity', 'A/G + C/C'): 'Negativa',
                        ('Toxicity', 'A/G + C/T'): 'Negativa',
                        ('Toxicity', 'A/A + C/C'): 'Intermedia',
                        ('Toxicity', 'G/G + T/T'): 'Intermedia',
                        ('Toxicity', 'G/G + C/T'): 'Intermedia',

                        # Metabolism/Pk (nueva imagen)
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
                    datos_cyp2c19.to_csv('datos_CYP2C19.tsv', sep='\t', index=False)
                    print("\nArchivo generado: datos_CYP2C19.tsv (expandido por genotipo)")
                    print("\nResumen de Evaluaciones para CYP2C19:")
                    print(datos_cyp2c19['Evaluacion'].value_counts(dropna=False))
                else:
                    print("Advertencia: No se encontró la columna 'genes' en los datos procesados para CYP2C19.")

                print("\n=== PROCESO COMPLETADO PARA CYP2D6 Y CYP2C19 ===")
            else:
                print("\nError: No se pudieron procesar los datos")
        else:
            print("\nError: No se pudo crear la columna Genotipo")
    else:
        print("\nError: No se pudieron cargar los datos")

if __name__ == "__main__":
    main() 