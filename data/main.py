from procesador_datos import ProcesadorDatos
from recomendador_ia import RecomendadorIA
import os

def validar_genotipo(genotipo: str, gen: str) -> bool:
    """
    Valida que el genotipo ingresado sea válido para el gen especificado.
    
    Args:
        genotipo: Genotipo a validar
        gen: Gen al que pertenece el genotipo ('CYP2D6' o 'CYP2C19')
    
    Returns:
        bool: True si el genotipo es válido, False en caso contrario
    """
    # Genotipos válidos para CYP2D6
    genotipos_d6 = ['G/G', 'G/A', 'A/A']
    # Genotipos válidos para CYP2C19
    genotipos_c19 = ['G/G + C/C', 'A/G + C/C', 'A/A + C/C', 'G/G + C/T', 'G/G + T/T', 'A/G + C/T']
    
    if gen == 'CYP2D6':
        return genotipo in genotipos_d6
    elif gen == 'CYP2C19':
        return genotipo in genotipos_c19
    return False

def obtener_genotipo(gen: str) -> str:
    """
    Solicita y valida el genotipo para un gen específico.
    
    Args:
        gen: Gen para el que se solicita el genotipo ('CYP2D6' o 'CYP2C19')
    
    Returns:
        str: Genotipo válido ingresado por el usuario
    """
    genotipos_validos = {
        'CYP2D6': ['G/G', 'G/A', 'A/A'],
        'CYP2C19': ['G/G + C/C', 'A/G + C/C', 'A/A + C/C', 'G/G + C/T', 'G/G + T/T', 'A/G + C/T']
    }
    
    while True:
        print(f"\nGenotipos válidos para {gen}:")
        for gt in genotipos_validos[gen]:
            print(f"- {gt}")
        
        genotipo = input(f"\nIngrese el genotipo para {gen}: ").strip()
        if validar_genotipo(genotipo, gen):
            return genotipo
        print(f"\nError: Genotipo inválido para {gen}. Por favor, ingrese uno de los genotipos válidos.")

def mostrar_recomendaciones(recomendaciones, titulo):
    """Muestra las recomendaciones de forma clara y detallada."""
    # Mapeo de evaluaciones a fenotipos según el tipo de evaluación y gen
    evaluacion_a_fenotipo = {
        'Positiva': {
            'CYP2D6': {
                'metabolizacion': 'Metabolizador Normal',
                'toxicidad': 'Baja toxicidad',
                'eficiencia': 'Alta eficiencia'
            },
            'CYP2C19': {
                'metabolizacion': 'Metabolizador Normal',
                'toxicidad': 'Baja toxicidad',
                'eficiencia': 'Alta eficiencia'
            }
        },
        'Intermedia': {
            'CYP2D6': {
                'metabolizacion': 'Metabolizador Intermedio',
                'toxicidad': 'Toxicidad moderada',
                'eficiencia': 'Eficiencia moderada'
            },
            'CYP2C19': {
                'metabolizacion': 'Metabolizador Intermedio',
                'toxicidad': 'Toxicidad moderada',
                'eficiencia': 'Eficiencia moderada'
            }
        },
        'Negativa': {
            'CYP2D6': {
                'metabolizacion': 'Metabolizador Lento',
                'toxicidad': 'Alta toxicidad',
                'eficiencia': 'Baja eficiencia'
            },
            'CYP2C19': {
                'metabolizacion': 'Metabolizador Lento',
                'toxicidad': 'Alta toxicidad',
                'eficiencia': 'Baja eficiencia'
            }
        }
    }
    
    # Función para determinar el tipo de evaluación basado en el fármaco y gen
    def determinar_tipo_evaluacion(farmaco, gen, evaluacion):
        # Por defecto, asumimos metabolización
        tipo = 'metabolizacion'
        
        # Lista de fármacos donde la toxicidad es el factor principal
        farmacos_toxicidad = ['Clozapina', 'Haloperidol', 'Risperidona']
        # Lista de fármacos donde la eficiencia es el factor principal
        farmacos_eficiencia = ['Fluoxetina', 'Paroxetina', 'Sertralina']
        
        if farmaco in farmacos_toxicidad:
            tipo = 'toxicidad'
        elif farmaco in farmacos_eficiencia:
            tipo = 'eficiencia'
            
        return tipo
    
    print(f"\n{titulo}")
    print("=" * 80)
    
    for i, rec in enumerate(recomendaciones, 1):
        print(f"\n{i}. {rec['farmaco']}")
        print(f"   Puntuación final: {rec['puntuacion']:.2f}")
        print(f"   Porcentaje de éxito: {rec['porcentaje_exito']:.1f}%")
        
        # Mostrar resumen de genes donde se encontró el fármaco
        genes_presentes = []
        if rec['presente_en']['CYP2C19']:
            genes_presentes.append('CYP2C19')
        if rec['presente_en']['CYP2D6']:
            genes_presentes.append('CYP2D6')
        print(f"   Presente en: {', '.join(genes_presentes)}")
        
        # Mostrar detalles por gen
        print("\n   Detalles por gen:")
        for gen in genes_presentes:
            predicciones = rec['predicciones'].get(gen, {})
            
            if predicciones:
                print(f"\n   {gen}:")
                evaluacion = predicciones['evaluacion']
                tipo_eval = determinar_tipo_evaluacion(rec['farmaco'], gen, evaluacion)
                
                # Obtener el fenotipo específico para el gen y tipo de evaluación
                fenotipo = evaluacion_a_fenotipo.get(evaluacion, {}).get(gen, {}).get(tipo_eval, evaluacion)
                print(f"      Evaluación: {evaluacion} ({fenotipo})")
                print(f"      Confianza: {predicciones['confianza']:.2f}")
                
                # Indicar nivel de significancia según el p-value
                p_value = predicciones['p_value']
                if p_value < 0.001:
                    print("      Significancia: Máxima (p < 0.001)")
                elif p_value < 0.01:
                    print("      Significancia: Alta (p < 0.01)")
                elif p_value < 0.05:
                    print("      Significancia: Moderada Alta (p < 0.05)")
                elif p_value < 0.1:
                    print("      Significancia: Moderada (p < 0.1)")
                else:
                    print("      Significancia: Baja (p >= 0.1)")
                
                if predicciones['confianza'] < 0.5:
                    print("      ⚠️  Advertencia: Baja confianza en la predicción")
        
        print("-" * 80)

def main():
    """
    Función principal que ejecuta el flujo completo del programa
    """
    print("=== INICIO DEL PROCESAMIENTO DE DATOS Y ANÁLISIS DE IA ===")
    
    # Verificar si los archivos procesados ya existen
    archivos_procesados = ['datos_CYP2D6.tsv', 'datos_CYP2C19.tsv']
    archivos_existentes = all(os.path.exists(archivo) for archivo in archivos_procesados)
    
    if archivos_existentes:
        print("\nLos archivos de datos procesados ya existen:")
        for archivo in archivos_procesados:
            print(f"- {archivo}")
        print("\nSaltando el procesamiento de datos...")
    else:
        # Crear instancia de la clase ProcesadorDatos
        procesador = ProcesadorDatos()
        
        # Cargar datos de variantes (usando archivo original)
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
                    
                    if datos_expandidos is not None:
                        # Crear columna Evaluacion
                        print("\n5. Creando columna Evaluacion...")
                        datos_cyp2d6, datos_cyp2c19 = procesador.crear_columna_evaluacion()
                        
                        if datos_cyp2d6 is None or datos_cyp2c19 is None:
                            print("\nError: No se pudo crear la columna Evaluacion")
                            return
                    else:
                        print("\nError: No se pudieron expandir los genotipos")
                        return
                else:
                    print("\nError: No se pudieron procesar los datos")
                    return
            else:
                print("\nError: No se pudo crear la columna Genotipo")
                return
        else:
            print("\nError: No se pudieron cargar los datos")
            return
    
    # Inicializar el recomendador de IA
    print("\n=== INICIALIZANDO RECOMENDADOR DE IA ===")
    recomendador = RecomendadorIA()
    
    # Solicitar genotipos al usuario
    print("\n=== INGRESO DE GENOTIPOS ===")
    genotipo_d6 = obtener_genotipo('CYP2D6')
    genotipo_c19 = obtener_genotipo('CYP2C19')
    
    # Obtener recomendaciones
    print("\n=== GENERANDO RECOMENDACIONES ===")
    mejores, peores = recomendador.recomendar_farmacos(genotipo_c19, genotipo_d6)
    
    # Mostrar resultados
    mostrar_recomendaciones(mejores, "=== MEJORES RECOMENDACIONES ===")
    mostrar_recomendaciones(peores, "=== PEORES RECOMENDACIONES ===")

if __name__ == "__main__":
    main() 