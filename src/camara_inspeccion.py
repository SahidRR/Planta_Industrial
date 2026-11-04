import cv2
import numpy as np
from src.cinta_transportadora import CintaTransportadora

class CamaraInspeccion:
    def __init__(self, nombre_camara: str, cinta: CintaTransportadora, area_minima: int = 300):
        self.nombre_camara = nombre_camara
        self.cinta = cinta
        self.area_minima = area_minima


        self.objeto_detectado: bool = False
        self.forma_detectada: str | None = None

    def capturar(self, renderer, data):
        renderer.update_scene(data, camera=self.nombre_camara)
        frame_rgb = renderer.render()
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    def detectar_forma(self, frame_bgr):
        """
        Procesa un fotograma BGR para identificar figuras geométricas sobre la cinta.
        
        Parameters
        ----------
        frame_bgr : np.ndarray
            Imagen en formato BGR obtenida de la cámara.
            
        Returns
        -------
        forma_detectada : str | None
            Nombre de la figura ("cuadrado", "rectangulo", "circulo") o None si no se detecta objeto.
        imagen_resultado : np.ndarray
            Fotograma anotado con contornos, cajas delimitadoras y texto.
        """
        imagen_resultado = frame_bgr.copy()

        # 1. Convertir de BGR a espacio HSV
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        # 2. Filtrar objetos coloreados frente al fondo de la cinta
        mask = cv2.inRange(
            hsv,
            np.array([0, 60, 40]),
            np.array([179, 255, 255])
        )

        # 3. Limpiar ruido mediante operaciones morfológicas
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 4. Encontrar contornos en la máscara
        contornos, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        forma_detectada = None

        # 5. Analizar cada contorno encontrado
        for contorno in contornos:
            area = cv2.contourArea(contorno)

            # Descartar contornos menores al área mínima configurada en la cámara
            if area < self.area_minima:
                continue

            perimetro = cv2.arcLength(contorno, True)
            if perimetro <= 0:
                continue

            # Aproximación poligonal para contar vértices
            aproximacion = cv2.approxPolyDP(contorno, 0.03 * perimetro, True)
            vertices = len(aproximacion)

            # Bounding box y relación de aspecto (ancho / alto)
            x, y, w, h = cv2.boundingRect(aproximacion)
            relacion_aspecto = w / float(h)

            # Circularidad (1.0 = círculo perfecto)
            circularidad = 4.0 * np.pi * area / (perimetro * perimetro)

            # Clasificación geométrica
            if vertices == 4:
                if 0.85 <= relacion_aspecto <= 1.15:
                    forma = "cuadrado"
                else:
                    forma = "rectangulo"
            elif vertices >= 6 and circularidad > 0.72:
                forma = "circulo"
            else:
                if circularidad > 0.78:
                    forma = "circulo"
                else:
                    forma = "desconocido"

            if forma != "desconocido":
                forma_detectada = forma

            # Centro de masa del objeto
            momentos = cv2.moments(contorno)
            if momentos["m00"] != 0:
                cx = int(momentos["m10"] / momentos["m00"])
                cy = int(momentos["m01"] / momentos["m00"])
            else:
                cx = x + w // 2
                cy = y + h // 2

            # Anotar visualmente la imagen para OpenCV
            cv2.drawContours(imagen_resultado, [contorno], -1, (0, 255, 0), 2)
            cv2.rectangle(imagen_resultado, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv2.circle(imagen_resultado, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(
                imagen_resultado,
                forma.upper(),
                (x, max(y - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        return forma_detectada, imagen_resultado

    def actualizar(self, renderer, data):
        frame_bgr = self.capturar(renderer, data)
        forma, frame_anotado = self.detectar_forma(frame_bgr)

        self.forma_detectada = forma
        self.objeto_detectado = forma is not None

        if self.objeto_detectado:
            self.cinta.detener()

        return frame_anotado