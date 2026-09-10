import numpy as np
import cv2
import matplotlib.pyplot as plt

def taller_analitico_1():
    print("=== TALLER ANALÍTICO 1: OPERACIONES CON TENSORES ===")
    
    # 1. Dimensiones de recortar imagen[100:200, 300:400, 1] en una imagen de 1920x1080
    # Filas: 200 - 100 = 100, Columnas: 400 - 300 = 100, Canal seleccionado: 1 (un solo canal)
    print("1. Dimensiones exactas (shape) de 'recorte':")
    print("   - Shape: (100, 100)")
    print("   - Contenido: Matriz 2D que contiene únicamente los valores de intensidad")
    print("     del canal Verde (G) en esa región de la imagen.\n")
    
    # 2. Eficiencia del Slicing vs Ciclos For
    print("2. ¿Por qué es más eficiente Slicing que ciclos 'for' anidados?:")
    print("   - Slicing aprovecha la contigüidad de memoria y vectorización en C/C++ de NumPy/OpenCV.")
    print("   - Evita la sobrecarga del intérprete de Python al recorrer píxel por píxel,")
    print("     permitiendo operaciones SIMD (Single Instruction, Multiple Data) en la CPU/GPU.\n")
    print("=" * 60 + "\n")


def taller_laboratorio_1():
    print("=== TALLER DE LABORATORIO 1: TRANSFORMACIÓN DE ESPACIOS ===")
    
    # 1. Crear un píxel BGR de prueba completamente amarillo intenso [0, 255, 255]
    pixel_bgr = np.array([0, 255, 255], dtype=np.float32) # B=0, G=255, R=255
    
    # 2 y 3. Cálculo matemático ponderado Y = 0.299*R + 0.587*G + 0.114*B (en BGR: 0.114*B + 0.587*G + 0.299*R)
    pesos = np.array([0.114, 0.587, 0.299])
    valor_gris = np.dot(pixel_bgr, pesos)
    
    print(f"Píxel BGR original (Amarillo): {pixel_bgr}")
    print(f"Valor de intensidad en escala de grises calculado: {valor_gris:.2f}")
    print("Respuesta: El amarillo puro arroja una intensidad luminosa alta (~225.93 / 255)")
    print("debido a la gran sensibilidad del ojo humano al canal verde y rojo.\n")
    
    # 4. Nota sobre conversión con OpenCV
    print("Para aplicar en imagen real con OpenCV:")
    print("   img_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)\n")
    print("=" * 60 + "\n")


def taller_laboratorio_2():
    print("=== TALLER DE LABORATORIO 2: ANÁLISIS ESTADÍSTICO ===")
    
    # Simulación de carga de una imagen RGB/BGR (creando un tensor sintético de 100x100x3)
    # Reemplazar 'muestra.jpg' por la ruta de tu imagen real
    np.random.seed(42)
    imagen = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    # Separa los 3 canales
    b, g, r = cv2.split(imagen)
    
    # Calcular histogramas para cada canal
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
    
    # Graficar los tres histogramas superpuestos
    plt.figure(figsize=(8, 5))
    plt.plot(hist_b, color='blue', label='Canal Azul (B)')
    plt.plot(hist_g, color='green', label='Canal Verde (G)')
    plt.plot(hist_r, color='red', label='Canal Rojo (R)')
    plt.title('Histograma de Canales de Color (BGR)')
    plt.xlabel('Intensidad de Píxel (0 - 255)')
    plt.ylabel('Cantidad de Píxeles')
    plt.legend()
    plt.grid(True)
    
    print("Histograma generado correctamente. Visualice el gráfico en pantalla.")
    print("Conclusión: El canal cuyo histograma se desplace más a la derecha (valores cercanos a 255)")
    print("es el predominantemente dominante en la iluminación general de la fotografía.")
    
    # Mostrar gráfico
    plt.show()


if __name__ == "__main__":
    taller_analitico_1()
    taller_laboratorio_1()
    taller_laboratorio_2()