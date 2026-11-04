class Horno:
    """
    Representa la estación industrial de horneado.
    Encapsula el conteo de piezas, el control de temperatura y su mapa visual de color en MuJoCo.
    """
    def __init__(self, geom_id: int = -1, temperatura_inicial: float = 20.0, temp_min: float = 0.0, temp_max: float = 200.0):
        self.geom_id = geom_id
        self.temperatura = temperatura_inicial
        self.temperatura_minima = temp_min
        self.temperatura_maxima = temp_max
        self.numero_piezas = 0

    def registrar_pieza(self) -> None:
        """Registra la entrada de una nueva pieza en el horno."""
        self.numero_piezas += 1

    def aumentar_temperatura(self, incremento: float) -> None:
        """Incrementa la temperatura del horno sin superar el límite máximo."""
        self.temperatura = min(self.temperatura + incremento, self.temperatura_maxima)

    def disminuir_temperatura(self, decremento: float) -> None:
        """Disminuye la temperatura del horno sin descender del límite mínimo."""
        self.temperatura = max(self.temperatura - decremento, self.temperatura_minima)

    def obtener_temperatura(self) -> float:
        """Devuelve la temperatura actual en grados Celsius."""
        return self.temperatura

    def actualizar_color(self, model) -> None:
        """
        Modifica el color RGBA de la geometría en MuJoCo en función de la temperatura actual:
            - Azul para temperatura baja (< 50 °C)
            - Naranja para temperatura media (50 °C - 100 °C)
            - Rojo para temperatura alta (> 100 °C)
        """
        if self.geom_id < 0 or model is None:
            return

        if self.temperatura < 50.0:
            color = [0.1, 0.3, 1.0, 1.0]   # Azul
        elif self.temperatura < 100.0:
            color = [1.0, 0.6, 0.1, 1.0]   # Naranja
        else:
            color = [1.0, 0.1, 0.1, 1.0]   # Rojo

        model.geom_rgba[self.geom_id] = color