import numpy as np


def taller_analitico_1():
    print("TALLER ANALÍTICO 1: INDEXACIÓN Y TENSORES")
    print(
        "1. Valor exacto del elemento A[2,3] en la Matriz A (5x5) con base 0:"
    )
    print("   Respuesta: 0 el valor representa ausencia de intensidad de luz (negro).")
    print(
        "   Visualmente corresponde a un píxel oscuro en el área interna cercana al centro.\n"
    )

    print("2. Bytes almacenados para un Tensor RGB (1080 x 1920 x 3):")
    total_elementos = 1080 * 1920 * 3
    megabytes = total_elementos / (1024 * 1024)
    print(
        f"   - Total de elementos/bytes (uint8): {total_elementos:,} bytes"
    )
    print(f"   - Tamaño aproximado en memoria: {megabytes:.2f} MB\n")
    


def taller_laboratorio_1():
    print("TALLER DE LABORATORIO 1: TRANSFORMACIONES AFINES")

    print("1. Matriz 5x5 simulando radiografía sobreexpuesta valores entre 200 y 255")
    np.random.seed(42)
    original = np.random.randint(200, 255, (5, 5))

    print("2. Reducir contraste al 50% (alpha = 0.5) y brillo en 50 unidades (beta = -50)") 
    alpha = 0.5
    beta = -50.0
    nueva = alpha * original + beta

    print("3. Truncar valores en [0, 255] y convertir a uint8\n")
    nueva = np.clip(nueva, 0, 255).astype(np.uint8)

    print("Matriz Original (Sobreexpuesta):\n", original)
    print("\nMatriz nueva (Ajuste de Brillo y Contraste):\n", nueva, "\n")


def taller_analitico_2():
    print("TALLER ANALÍTICO 2: TRANSFORMACIONES")
    print("1. Transposición de una Matriz Identidad I_4 (4x4):")
    print("   Respuesta: La matriz resultante sigue siendo I_4 (I^T = I).")
    print(
        "   Explicación: Es una matriz simétrica con 1s en la diagonal principal."
    )
    print(
        "   Al intercambiar filas por columnas, los 1s no se mueven y los 0s intercambian posición entre sí.\n"
    )

    print("2. Neuronas necesarias para aplanar un tensor RGB de (200, 200, 3):")
    neuronas = 200 * 200 * 3
    print(
        f"   - Cálculo: 200 x 200 x 3 = {neuronas:,} neuronas en la capa de entrada.\n"
    )


def taller_laboratorio_final():
    print("TALLER DE LABORATORIO FINAL: PROGRAMANDO UN KERNEL")

    print("1. Matriz 'Sección de Imagen (I)' y 'Kernel (K)'")
    I = np.array(
        [[100, 100, 100], [100, 200, 100], [100, 100, 100]], dtype=np.float32
    )

    K = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)

    print("2. Producto Hadamard (elemento a elemento) y suma total")
    producto_hadamard = I * K
    pixel_central = np.sum(producto_hadamard)

    print("Sección de Imagen (I):\n", I)
    print("\nKernel de Realce (K):\n", K)
    print("\nProducto Hadamard (I * K):\n", producto_hadamard)
    print(
        f"\nValor resultante para el píxel central procesado: {pixel_central}\n"
    )
    


if __name__ == "__main__":
    taller_analitico_1()
    taller_laboratorio_1()
    taller_analitico_2()
    taller_laboratorio_final()