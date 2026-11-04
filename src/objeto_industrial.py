import numpy as np
 
class ObjetoIndustrial:
    """
    Super clase ase para todos los objetos y piezas de la planta industrial.
    Encapsula el estado común
    """
    def __init__(self, id_objeto: int, nombre_body: str, forma: str, body_id: int = -1, segmento: int = 0):
        self.id_objeto = id_objeto
        self.nombre_body = nombre_body
        self.forma = forma
        self.body_id = body_id
        self.segmento = segmento
        self.posicion = np.zeros(3, dtype=float)
    def actualizar_posicion(self, data) -> np.ndarray:
        """Actualiza y devuelve la posición real de la pieza desde MuJoCo."""
        if self.body_id >= 0:
            self.posicion = data.xpos[self.body_id].copy()
        return self.posicion
    def actualizar_segmento(self, nuevo_segmento: int) -> None:
        """Actualiza el segmento de cinta en el que transita el objeto."""
        self.segmento = nuevo_segmento
    def obtener_pose(self, data) -> np.ndarray:
        """Devuelve las coordenadas 3D actuales de la pieza."""
        return self.actualizar_posicion(data)

class Caja(ObjetoIndustrial):
    """
    Subclase que representa piezas con geometría cúbica / rectangular.
    """
    def __init__(self, id_objeto: int, nombre_body: str, body_id: int = -1, segmento: int = 0):
        # Llama al constructor de la superclase fijando la forma como "caja"
        super().__init__(id_objeto, nombre_body, forma="caja", body_id=body_id, segmento=segmento)

class Cilindro(ObjetoIndustrial):
    """
    Subclase que representa piezas cilíndricas (ej. objetos azul y verde).
    """
    def __init__(self, id_objeto: int, nombre_body: str, body_id: int = -1, segmento: int = 0):
        # Llama al constructor de la superclase fijando la forma como "cilindro"
        super().__init__(id_objeto, nombre_body, forma="cilindro", body_id=body_id, segmento=segmento)
