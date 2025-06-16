from procesador_datos import ProcesadorDatos
from recomendador_ia import RecomendadorIA
import pandas as pd
import re
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
    print(f"\n{titulo}")
    print("=" * 80)
    
    for i, rec in enumerate(recomendaciones, 1):
        print(f"\n{i}. {rec['farmaco']}")
        print(f"   Puntuación final: {rec['puntuacion']:.2f}")
        
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
                # Calcular evaluación más común y confianza promedio
                evaluaciones = []
                confianzas = []
                p_values = []
                for modelo, pred in predicciones.items():
                    if pred and 'evaluacion' in pred and 'confianza' in pred:
                        evaluaciones.append(pred['evaluacion'])
                        confianzas.append(pred['confianza'])
                        p_values.append(pred.get('p_value', 0.5))
                
                if evaluaciones:
                    eval_mas_comun = max(set(evaluaciones), key=evaluaciones.count)
                    confianza_promedio = sum(confianzas) / len(confianzas)
                    p_value_promedio = sum(p_values) / len(p_values)
                    
                    print(f"\n   {gen}:")
                    print(f"      Evaluación: {eval_mas_comun}")
                    print(f"      Confianza promedio: {confianza_promedio:.2f}")
                    print(f"      P-value promedio: {p_value_promedio:.3f}")
                    
                    # Indicar nivel de significancia según la nueva escala
                    if p_value_promedio < 0.001:
                        print("      Significancia: Máxima (p < 0.001)")
                    elif p_value_promedio < 0.01:
                        print("      Significancia: Alta (p < 0.01)")
                    elif p_value_promedio < 0.05:
                        print("      Significancia: Moderada Alta (p < 0.05)")
                    elif p_value_promedio < 0.1:
                        print("      Significancia: Moderada (p < 0.1)")
                    else:
                        print("      Significancia: Baja (p >= 0.1)")
                    
                    if confianza_promedio < 0.5:
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
    
    # Inicializar y entrenar el recomendador de IA
    print("\n=== INICIALIZANDO RECOMENDADOR DE IA ===")
    recomendador = RecomendadorIA()
    
    # Verificar si ya existen modelos entrenados
    try:
        recomendador.cargar_modelos()
        print("\nModelos cargados exitosamente")
    except:
        print("\nEntrenando nuevos modelos...")
        recomendador.entrenar_modelos()
    
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