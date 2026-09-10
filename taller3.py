import cv2
import numpy as np


def taller_analitico_1():
    print("=== TALLER ANALÍTICO: UMBRALIZACIÓN Y MORFOLOGÍA ===")

    print("1. Método de Otsu vs. Umbralización Estática:")
    print(
        "   - Umbralización Estática: Requiere fijar manualmente un valor de umbral (T)."
    )
    print(
        "     Es ineficiente si las condiciones de iluminación o el contraste cambian."
    )
    print(
        "   - Método de Otsu: Calcula automáticamente el umbral óptimo analizando"
    )
    print(
        "     el histograma de la imagen y minimizando la varianza intra-clase.\n"
    )

    print("2. Operaciones Morfológicas (Apertura vs. Cierre):")
    print(
        "   - Apertura (Erosión seguida de Dilatación): Elimina pequeño ruido externo"
    )
    print(
        "     o protuberancias finas sin alterar significativamente el tamaño del objeto."
    )
    print(
        "   - Cierre (Dilatación seguida de Erosión): Rellena pequeños huecos internos"
    )
    print("     o grietas dentro de los objetos segmentados.\n")
    print("=" * 60 + "\n")


def taller_laboratorio_1():
    print("=== TALLER DE LABORATORIO 1: SEGMENTACIÓN Y BINARIZACIÓN ===")

    # 1. Crear una imagen sintética en escala de grises con fondo oscuro y un objeto brillante
    np.random.seed(42)
    imagen = np.full((100, 100), 50, dtype=np.uint8)  # Fondo
    imagen[30:70, 30:70] = 200  # Objeto central

    # Agregar algo de ruido sintético
    ruido = np.random.randint(0, 30, (100, 100), dtype=np.uint8)
    imagen = cv2.add(imagen, ruido)

    # 2. Aplicar Binarización Estática (T = 127)
    _, img_binaria_estatica = cv2.threshold(imagen, 127, 255, cv2.THRESH_BINARY)

    # 3. Aplicar Binarización de Otsu
    t_otsu, img_otsu = cv2.threshold(
        imagen, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    print("Resultados de Segmentación:")
    print("   - Umbral fijo aplicado: 127")
    print(f"   - Umbral óptimo calculado por Otsu: {t_otsu}")
    print("\n" + "=" * 60 + "\n")


def taller_laboratorio_2():
    print("=== TALLER DE LABORATORIO 2: OPERACIONES MORFOLÓGICAS ===")

    # 1. Crear imagen binaria sintética con ruido (puntos blancos aislados)
    img_binaria = np.zeros((100, 100), dtype=np.uint8)
    img_binaria[30:70, 30:70] = 255  # Objeto principal

    # Ruido externo (puntos pequeños de ruido)
    img_binaria[10, 10] = 255
    img_binaria[15, 80] = 255
    img_binaria[85, 20] = 255

    # 2. Definir el Elemento Estructurante (Kernel) de 5x5 de unos
    kernel = np.ones((5, 5), np.uint8)

    # 3. Aplicar Erosión (Elimina ruido externo)
    img_erosionada = cv2.erode(img_binaria, kernel, iterations=1)

    # 4. Aplicar Dilatación sobre la imagen erosionada (Rellena/restaura el objeto)
    img_dilatada = cv2.dilate(img_erosionada, kernel, iterations=1)

    # Nota: Erosión + Dilatación = Operación de Apertura (Opening)
    img_apertura = cv2.morphologyEx(img_binaria, cv2.MORPH_OPEN, kernel)

    print("Morfología Matemática Procesada:")
    print("   - Kernel utilizado: Matriz de 5x5 (np.ones)")
    print(
        "   - Operación de Apertura ejecutada exitosamente (Ruido externo eliminado)."
    )
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    taller_analitico_1()
    taller_laboratorio_1()
    taller_laboratorio_2()