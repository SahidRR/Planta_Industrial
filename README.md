
# Planta Industrial Automática - Refactorización POO

Práctica I de la asignatura Programación Modular y Orientada a Objetos, del Grado en Tecnologías Digitales para la Empresa (Universidad de Burgos).

El objetivo de este proyecto es coger una simulación industrial desarrollada con el motor físico MuJoCo y la librería OpenCV, y refactorizar su código. Se ha pasado de un script procedural monolítico a una arquitectura basada en Programación Orientada a Objetos (POO), buscando un código más mantenible y estructurado.

## Descripción de la simulación

La simulación muestra un entorno industrial automatizado que incluye:

- Un circuito cerrado formado por 4 cintas transportadoras principales.
- Una cinta lateral que conecta con un horno industrial de calentamiento continuo.
- Una estación de inspección visual (usando OpenCV). Detecta y clasifica las piezas según su geometría (cuadrados, rectángulos, círculos) y detiene la cinta automáticamente cuando corresponde.
- Un brazo robótico (Franka Emika Panda) que realiza tareas de pick-and-place automatizadas usando cinemática inversa (DLS).
- Un horno industrial dinámico con control interactivo de temperatura y cambios de iluminación RGBA en tiempo real.

## Estructura del repositorio

El código separa el script de ejecución de la lógica del dominio, que está dentro de la carpeta `src/`:

```text
planta_industrial/
├── main.py                      # Script de ejecución (Composition Root)
├── planta_4_cintas.py           # Generador procedural del modelo físico MJCF
├── robot.py                     # Solver cinemático DLS del brazo robótico
├── planta_industrial.xml        # Modelo de la planta en MuJoCo
├── .gitignore                   
├── README.md                    
└── src/                         # Clases del dominio
    ├── __init__.py              
    ├── cinta_transportadora.py  # Clase CintaTransportadora
    ├── camara_inspeccion.py     # Clase CamaraInspeccion (Lógica de OpenCV)
    ├── objeto_industrial.py     # Clases ObjetoIndustrial, Caja y Cilindro
    ├── horno.py                 # Clase Horno
    ├── robot_panda.py           # Clase RobotPanda
    └── planta.py                # Clase PlantaIndustrial
```

## Principios POO aplicados

Para llevar a cabo la refactorización, se han aplicado los siguientes conceptos:

- **Encapsulación:** Se han eliminado las variables globales (como las antiguas banderas de las cintas). Ahora, el estado de marcha o paro es privado dentro de cada objeto `CintaTransportadora` y `Horno`, y solo se modifica a través de sus propios métodos (`arrancar()`, `detener()`, `conmutar()`).
- **Abstracción:** Toda la complejidad matemática del procesamiento de imagen (BGR/HSV) y los cálculos del robot en MuJoCo se han abstraído en métodos simples como `camara.actualizar()` o `robot.actualizar_ik()`.
- **Herencia y Polimorfismo:** Las clases `Caja` y `Cilindro` heredan de una clase base `ObjetoIndustrial`. De esta forma, la clase `PlantaIndustrial` puede llamar a `actualizar_posicion()` iterando sobre una lista de objetos sin importar si son cajas o cilindros.
- **Composition Root:** La clase `PlantaIndustrial` actúa como núcleo, ensamblando las dependencias y coordinando la simulación. Esto permite que el archivo `main.py` quede totalmente limpio y solo se encargue de arrancar el programa.

## Instalación y uso

Es necesario tener Python 3.10 o superior. 

1. Clona el repositorio:
```bash
git clone https://github.com/tu-usuario/planta_industrial.git
cd planta_industrial
```

2. (Opcional) Crea un entorno virtual:
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
```

3. Instala las dependencias requeridas:
```bash
pip install mujoco opencv-python numpy
```

4. Ejecuta la simulación:
```bash
python main.py
```

## Licencia

Proyecto desarrollado con fines académicos para la Universidad de Burgos (UBU). Consulta el archivo `LICENSE` para más información.