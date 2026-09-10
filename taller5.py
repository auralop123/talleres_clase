import cv2
import numpy as np


def taller_analitico_1():
    print("=== TALLER ANALÍTICO: CALCULANDO EL GRADIENTE DE SOBEL ===")

    print(
        "1. Convolución de la matriz 3x3 con Kernel Sobel X (Gx) en el píxel central:"
    )
    print("   Matriz Imagen I:")
    print("   [[  0,   0, 255],")
    print("    [  0,   0, 255],")
    print("    [  0,   0, 255]]\n")

    print("   Kernel Sobel X (Gx):")
    print("   [[-1, 0, 1],")
    print("    [-2, 0, 2],")
    print("    [-1, 0, 1]]\n")

    print("   Cálculo elemental punto a punto:")
    print("   Gx = (0*-1) + (0*0) + (255*1) +")
    print("        (0*-2) + (0*0) + (255*2) +")
    print("        (0*-1) + (0*0) + (255*1)")
    print("   Gx = 255 + 510 + 255 = 1020")
    print("   Valor del Gradiente en X (Gx): 1020 (o 255 en escala normalizada)\n")

    print(
        "2. Convolución con Kernel Sobel Y (Gy) y dirección del borde:"
    )
    print("   Kernel Sobel Y (Gy):")
    print("   [[-1, -2, -1],")
    print("    [ 0,  0,  0],")
    print("    [ 1,  2,  1]]\n")

    print("   Cálculo elemental punto a punto:")
    print("   Gy = (0*-1) + (0*-2) + (255*-1) +")
    print("        (0*0)  + (0*0)  + (255*0)  +")
    print("        (0*1)  + (0*2)  + (255*1)")
    print("   Gy = -255 + 0 + 255 = 0")
    print(
        "   Explicación: El resultado es 0 porque no hay variación de intensidad vertical."
    )
    print(
        "   Esto indica que el borde detectado es completamente VERTICAL (cambio en el eje horizontal X).\n"
    )
    print("=" * 60 + "\n")


def taller_laboratorio_1():
    print("=== TALLER DE LABORATORIO: SOBEL Y DETECTOR DE BORDES CANNY ===")

    # 1. Crear una imagen sintética de prueba (cuadrado blanco centrado)
    imagen = np.zeros((100, 100), dtype=np.uint8)
    imagen[25:75, 25:75] = 255

    # 2. Aplicar Operador de Sobel en X y en Y (usando CV_64F para evitar desbordamientos)
    sobel_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)

    # 3. Calcular Magnitud del Gradiente (Magnitud = sqrt(Gx^2 + Gy^2))
    magnitud = cv2.magnitude(sobel_x, sobel_y)
    magnitud_8u = np.uint8(np.clip(magnitud, 0, 255))

    # 4. Aplicar Detector de Bordes Canny (Umbrales: 100 y 200)
    bordes_canny = cv2.Canny(imagen, 100, 200)

    print("Procesamiento de Bordes Finalizado:")
    print("   - Sobel X (Bordes verticales) y Sobel Y (Bordes horizontales) calculados.")
    print("   - Magnitud del Gradiente obtenida exitosamente.")
    print("   - Detector de Canny ejecutado (Supresión de No-Máximos y Umbralización con Histéresis).")
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    taller_analitico_1()
    taller_laboratorio_1()