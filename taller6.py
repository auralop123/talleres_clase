import cv2
import numpy as np


def taller_analitico_1():
    print("=== TALLER ANALÍTICO: CONTORNOS Y CARACTERÍSTICAS ===")

    print("1. RETR_EXTERNAL vs. RETR_TREE:")
    print(
        "   - RETR_EXTERNAL: Recupera únicamente los contornos externos (exteriores)."
    )
    print(
        "     Ignora por completo las jerarquías y agujeros dentro de los objetos."
    )
    print(
        "   - RETR_TREE: Recupera todos los contornos y reconstruye una jerarquía"
    )
    print(
        "     completa en forma de árbol (padres, hijos y huecos internos).\n"
    )

    print("2. CHAIN_APPROX_SIMPLE vs. CHAIN_APPROX_NONE:")
    print(
        "   - CHAIN_APPROX_NONE: Almacena absolutamente todos los puntos del contorno."
    )
    print(
        "   - CHAIN_APPROX_SIMPLE: Comprime segmentos horizontales, verticales y diagonales,"
    )
    print(
        "     almacenando solo sus puntos extremos (p. ej., un rectángulo usa solo 4 puntos)."
    )
    print(
        "     Esto ahorra memoria y reduce significativamente el tiempo de procesamiento.\n"
    )

    print("3. Cálculo de Redondez / Circularidad:")
    print("   - Fórmula: Circularidad = (4 * pi * Área) / (Perímetro ^ 2)")
    print(
        "   - Un círculo perfecto arroja un valor cercano a 1.0; valores menores indican formas alargadas o irregulares.\n"
    )
    print("=" * 60 + "\n")


def taller_laboratorio_1():
    print("=== TALLER DE LABORATORIO: ANÁLISIS DE OBJETOS Y CARACTERÍSTICAS ===")

    # 1. Crear imagen binaria sintética con objetos blancos sobre fondo negro
    imagen_binaria = np.zeros((200, 200), dtype=np.uint8)

    # Dibujar un cuadrado y un círculo
    cv2.rectangle(imagen_binaria, (20, 20), (80, 80), 255, -1)
    cv2.circle(imagen_binaria, (140, 140), 30, 255, -1)

    # Crear versión BGR a color para dibujar resultados
    imagen_color = cv2.cvtColor(imagen_binaria, cv2.COLOR_GRAY2BGR)

    # 2. Encontrar contornos
    contornos, _ = cv2.findContours(
        imagen_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    print(f"Total de objetos/contornos detectados: {len(contornos)}\n")

    # 3. Iterar y extraer métricas geométricas
    for i, cnt in enumerate(contornos, start=1):
        area = cv2.contourArea(cnt)
        perimetro = cv2.arcLength(cnt, True)

        # Filtrar ruido pequeño
        if area > 100:
            # Bounding Box
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(imagen_color, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Centroide usando Momentos de Inercia
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(imagen_color, (cx, cy), 4, (0, 0, 255), -1)

            # Redondez / Circularidad
            circularidad = (4 * np.pi * area) / (perimetro**2) if perimetro > 0 else 0

            print(f"Objeto #{i}:")
            print(f"   - Área: {area:.1f} px")
            print(f"   - Perímetro: {perimetro:.1f} px")
            print(f"   - Centroide: ({cx}, {cy})")
            print(f"   - Bounding Box: [x={x}, y={y}, w={w}, h={h}]")
            print(f"   - Circularidad / Redondez: {circularidad:.2f}")
            print("-" * 40)

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    taller_analitico_1()
    taller_laboratorio_1()