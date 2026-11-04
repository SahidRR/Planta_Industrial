"""
Punto de entrada principal de la aplicación.
Instancia la Raíz de Composición (PlantaIndustrial) y arranca la simulación.
"""
from src.planta import PlantaIndustrial

def main():
    # Ruta al archivo del modelo de la simulación en MuJoCo
    xml_path = "planta_industrial.xml"
    
    # Instanciación de la planta y ejecución
    planta = PlantaIndustrial(xml_path)
    planta.ejecutar()

if __name__ == "__main__":
    main()