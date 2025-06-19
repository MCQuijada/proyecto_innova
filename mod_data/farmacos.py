import json

# Cargar el JSON con varias recomendaciones
with open("recomendaciones.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Extraer los nombres de los fármacos
nombres_farmacos = [r["farmaco"] for r in data]

# Guardarlos en un archivo de texto
with open("farmacos.txt", "w", encoding="utf-8") as f_out:
    for nombre in nombres_farmacos:
        f_out.write(nombre + "\n")

print("✅ Fármacos guardados en 'farmacos.txt'")
