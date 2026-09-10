import cv2
import numpy as np


def taller_analitico_1():
    print("=== TALLER ANALÍTICO 1: FILTROS ESPACIALES Y CONVOLUCIÓN ===")

    print("1. Filtro Gaussiano vs. Filtro de Media:")
    print(
        "   - Filtro de Media: Asigna el mismo peso a todos los píxeles dentro del kernel."
    )
    print(
        "     Esto borra o difumina los bordes reales de los objetos al promediar de manera uniforme."
    )
    print(
        "   - Filtro Gaussiano: Aplica pesos basados en la distribución gaussiana,"
    )
    print(
        "     dando más importancia al píxel central y menos a los más alejados."
    )
    print("     Suaviza el ruido conservando mejor las estructuras y bordes.\n")

    print("2. Naturaleza del Filtro de Mediana:")
    print(
        "   - No realiza una convolución matemática ni multiplicaciones matriciales."
    )
    print(
        "   - Ordena los valores bajo la ventana del kernel de menor a mayor"
    )
    print("     y reemplaza el píxel central por el valor de la mediana.")
    print(
        "   - Es ideal para eliminar ruido impulsivo ('sal y pimienta') sin distorsionar bordes.\n"
    )
    print("=" * 60 + "\n")


def taller_laboratorio_1():
    print("=== TALLER DE LABORATORIO: ESTRATEGIAS DE SUAVIZADO ===")

    # 1. Crear una imagen de prueba limpia (100x100 en escala de grises con fondo gris)
    imagen_limpia = np.full((100, 100), 128, dtype=np.uint8)

    # Añadir ruido sintético de "Sal y Pimienta"
    imagen_ruidosa = imagen_limpia.copy()
    np.random.seed(42)

    # 5% de píxeles negros (pimienta - 0) y 5% de píxeles blancos (sal - 255)
    num_ruido = int(0.05 * imagen_limpia.size)

    # Sal
    coords_sal = [
        np.random.randint(0, i, num_ruido) for i in imagen_limpia.shape
    ]
    imagen_ruidosa[tuple(coords_sal)] = 255

    # Pimienta
    coords_pimienta = [
        np.random.randint(0, i, num_ruido) for i in imagen_limpia.shape
    ]
    imagen_ruidosa[tuple(coords_pimienta)] = 0

    # 2. Aplicar los tres filtros con un Kernel de 7x7
    kernel_size = 7

    # Filtro de Media (Promedio simple)
    blur_media = cv2.blur(imagen_ruidosa, (kernel_size, kernel_size))

    # Filtro Gaussiano
    blur_gauss = cv2.GaussianBlur(
        imagen_ruidosa, (kernel_size, kernel_size), 0
    )

    # Filtro de Mediana
    blur_mediana = cv2.medianBlur(imagen_ruidosa, kernel_size)

    print("Procesamiento finalizado con Kernel de 7x7:")
    print("   - Filtro de Media aplicado: cv2.blur()")
    print("   - Filtro Gaussiano aplicado: cv2.GaussianBlur()")
    print("   - Filtro de Mediana aplicado: cv2.medianBlur()")
    print("\n" + "=" * 60 + "\n")

    print("=== ANÁLISIS CRÍTICO DE RESULTADOS ===")
    print(
        "• El Filtro de Mediana ignora por completo los puntos extremos (0 y 255) porque"
    )
    print(
        "  al ordenar numéricamente los valores del kernel, los picos de ruido terminan"
    )
    print(
        "  en los extremos de la lista y la mediana siempre selecciona un valor representativo del fondo."
    )
    print(
        "• El Filtro de Media genera 'manchas' grises alrededor del ruido porque los valores de"
    )
    print(
        "  0 y 255 se promedian numéricamente con los píxeles vecinos, alterando el valor"
    )
    print("  promedio de toda la región cubierta por el kernel.\n")


if __name__ == "__main__":
    taller_analitico_1()
    taller_laboratorio_1()