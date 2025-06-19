from procesador_datos import ProcesadorDatos
from recomendador_ia import RecomendadorIA
import os
import json

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
    Obtiene y valida el genotipo desde un archivo JSON para un gen específico.

    Args:
        gen: Gen para el que se solicita el genotipo ('CYP2D6' o 'CYP2C19')

    Returns:
        str: Genotipo válido leído desde el archivo JSON

    Raises:
        ValueError: Si el archivo no existe, está mal formado o el genotipo no es válido
    """
    genotipos_validos = {
        'CYP2D6': ['G/G', 'G/A', 'A/A'],
        'CYP2C19': ['G/G + C/C', 'A/G + C/C', 'A/A + C/C', 'G/G + C/T', 'G/G + T/T', 'A/G + C/T']
    }

    archivo_json = "genotipos.json"

    try:
        with open(archivo_json, "r", encoding="utf-8") as f:
            datos = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Archivo '{archivo_json}' no encontrado.")
    except json.JSONDecodeError:
        raise ValueError(f"El archivo '{archivo_json}' no tiene formato JSON válido.")

    if gen not in datos:
        raise ValueError(f"No se encontró el gen '{gen}' en el archivo.")

    genotipo = datos[gen].strip()

    if genotipo not in genotipos_validos[gen]:
        raise ValueError(f"Genotipo inválido para {gen}: '{genotipo}'.\nGenotipos válidos: {genotipos_validos[gen]}")

    return genotipo


def mostrar_recomendaciones(recomendaciones, titulo, solo_estructura=False):
    """Muestra las recomendaciones de forma clara y detallada y/o devuelve la estructura para guardar en json."""
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
    def determinar_tipo_evaluacion(farmaco, gen, evaluacion):
        tipo = 'metabolizacion'
        farmacos_toxicidad = ['Clozapina', 'Haloperidol', 'Risperidona']
        farmacos_eficiencia = ['Fluoxetina', 'Paroxetina', 'Sertralina']
        if farmaco in farmacos_toxicidad:
            tipo = 'toxicidad'
        elif farmaco in farmacos_eficiencia:
            tipo = 'eficiencia'
        return tipo

    resultados_json = []
    if not solo_estructura:
        print(f"\n{titulo}")
        print("=" * 80)
    for i, rec in enumerate(recomendaciones, 1):
        resultado = {
            'farmaco': rec['farmaco'],
            'puntuacion': rec['puntuacion'],
            'porcentaje_exito': rec['porcentaje_exito'],
            'presente_en': [],
            'detalles': {}
        }
        genes_presentes = []
        if rec['presente_en']['CYP2C19']:
            genes_presentes.append('CYP2C19')
        if rec['presente_en']['CYP2D6']:
            genes_presentes.append('CYP2D6')
        resultado['presente_en'] = genes_presentes
        detalles_gen = {}
        for gen in genes_presentes:
            predicciones = rec['predicciones'].get(gen, {})
            if predicciones:
                evaluacion = predicciones['evaluacion']
                tipo_eval = determinar_tipo_evaluacion(rec['farmaco'], gen, evaluacion)
                fenotipo = evaluacion_a_fenotipo.get(evaluacion, {}).get(gen, {}).get(tipo_eval, evaluacion)
                detalle = {
                    'evaluacion': evaluacion,
                    'fenotipo': fenotipo,
                    'confianza': predicciones['confianza'],
                    'p_value': predicciones['p_value']
                }
                detalles_gen[gen] = detalle
        resultado['detalles'] = detalles_gen
        resultados_json.append(resultado)
        if not solo_estructura:
            print(f"\n{i}. {rec['farmaco']}")
            print(f"   Puntuación final: {rec['puntuacion']:.2f}")
            print(f"   Porcentaje de éxito: {rec['porcentaje_exito']:.1f}%")
            print(f"   Presente en: {', '.join(genes_presentes)}")
            print("\n   Detalles por gen:")
            for gen in genes_presentes:
                predicciones = rec['predicciones'].get(gen, {})
                if predicciones:
                    print(f"\n   {gen}:")
                    evaluacion = predicciones['evaluacion']
                    tipo_eval = determinar_tipo_evaluacion(rec['farmaco'], gen, evaluacion)
                    fenotipo = evaluacion_a_fenotipo.get(evaluacion, {}).get(gen, {}).get(tipo_eval, evaluacion)
                    print(f"      Evaluación: {evaluacion} ({fenotipo})")
                    print(f"      Confianza: {predicciones['confianza']:.2f}")
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
    return resultados_json

def guardar_resultados_json(nombre_archivo, datos):
    """Guarda los datos en un archivo JSON con codificación UTF-8."""
    with open(nombre_archivo, 'w', encoding='utf-8') as f:
        json.dump(datos, f, ensure_ascii=False, indent=2)

def unificar_recomendaciones(recomendaciones):
    """Elimina duplicados de fármacos, dejando solo la mejor entrada por fármaco y unificando detalles de genes."""
    unificados = {}
    for rec in recomendaciones:
        nombre = rec['farmaco']
        if nombre not in unificados:
            unificados[nombre] = rec.copy()
        else:
            # Si ya existe, comparar puntuación y unificar detalles
            if rec['puntuacion'] > unificados[nombre]['puntuacion']:
                base = rec.copy()
                # Unificar detalles de genes
                for gen, det in unificados[nombre]['detalles'].items():
                    if gen not in base['detalles']:
                        base['detalles'][gen] = det
                base['presente_en'] = list(set(base['presente_en']) | set(unificados[nombre]['presente_en']))
                unificados[nombre] = base
            else:
                # Unificar detalles de genes
                for gen, det in rec['detalles'].items():
                    if gen not in unificados[nombre]['detalles']:
                        unificados[nombre]['detalles'][gen] = det
                unificados[nombre]['presente_en'] = list(set(unificados[nombre]['presente_en']) | set(rec['presente_en']))
    # Devolver lista ordenada por puntuación descendente
    return sorted(unificados.values(), key=lambda x: x['puntuacion'], reverse=True)

def main():
    """
    Función principal que ejecuta el flujo completo del programa
    """
    print("=== INICIO DEL PROCESAMIENTO DE DATOS Y ANÁLISIS DE IA ===")
    archivos_procesados = ['datos_CYP2D6.tsv', 'datos_CYP2C19.tsv']
    archivos_existentes = all(os.path.exists(archivo) for archivo in archivos_procesados)
    if archivos_existentes:
        print("\nLos archivos de datos procesados ya existen:")
        for archivo in archivos_procesados:
            print(f"- {archivo}")
        print("\nSaltando el procesamiento de datos...")
    else:
        procesador = ProcesadorDatos()
        print("\n1. Cargando datos de variantes...")
        datos_variante = procesador.cargar_datos('datos_variante.tsv')
        if datos_variante is not None:
            print("\n2. Creando columna Genotipo...")
            datos_con_genotipo = procesador.crear_columna_genotipo()
            if datos_con_genotipo is not None:
                print("\n3. Limpiando y procesando datos...")
                datos_procesados = procesador.limpiar_datos()
                if datos_procesados is not None:
                    print("\n4. Expandiendo genotipos...")
                    datos_expandidos = procesador.expandir_genotipos()
                    if datos_expandidos is not None:
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
    print("\n=== INICIALIZANDO RECOMENDADOR DE IA ===")
    recomendador = RecomendadorIA()
    print("\n=== INGRESO DE GENOTIPOS ===")
    genotipo_d6 = obtener_genotipo('CYP2D6')
    genotipo_c19 = obtener_genotipo('CYP2C19')
    print("\n=== GENERANDO RECOMENDACIONES ===")
    mejores, peores = recomendador.recomendar_farmacos(genotipo_c19, genotipo_d6)
    # Mostrar y guardar resultados generales
    mejores_json = mostrar_recomendaciones(mejores, "=== MEJORES RECOMENDACIONES ===")
    peores_json = mostrar_recomendaciones(peores, "=== PEORES RECOMENDACIONES ===")
    # Unificar recomendaciones para eliminar duplicados
    mejores_json = unificar_recomendaciones(mejores_json)
    peores_json = unificar_recomendaciones(peores_json)
    # Guardar y mostrar resultados como antes
    guardar_resultados_json("mejores_recomendaciones.json", mejores_json)
    guardar_resultados_json("peores_recomendaciones.json", peores_json)
    guardar_resultados_json("recomendaciones.json", mejores_json)
    top5_mejores = mejores_json[:5]
    top5_peores = mejores_json[-5:]
    guardar_resultados_json("top5_mejores_recomendaciones.json", top5_mejores)
    guardar_resultados_json("top5_peores_recomendaciones.json", top5_peores)

if __name__ == "__main__":
    main() 