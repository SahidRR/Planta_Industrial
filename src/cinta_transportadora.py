class CintaTransportadora:
    """ 
    Representa una cinta transportadira de la planta industrial.
    Encapsula el estada de marcha/paro y sus parametros operativos
    """
    def __init__(self, id_cinta: int, nombre: str, velocidad: float, activa: bool =True):
        self.id_cinta = id_cinta
        self.nombre = nombre
        self.velocidad = velocidad
        self.activa = activa
    def arrancar(self) -> None:
        # Pone la cinta en marcha
        self.activa = True
    def detener(self) -> None:
        # Detiene la cinta.
        self.activa = False
    def conmutar(self) -> bool:
        # Alterna el estado de marcha/paro de la cinta.
        self.activa = not self.activa
        return self.activa
    def esta_activa(self) -> bool:
        # Devuelve True si la cinta esta en marcha, False si esta detenida
        return self.activa
