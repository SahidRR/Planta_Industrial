import os
import time
import cv2
import mujoco
import mujoco.viewer
import numpy as np

# Importaciones de clases refactorizadas desde src
from src.cinta_transportadora import CintaTransportadora
from src.camara_inspeccion import CamaraInspeccion
from src.objeto_industrial import ObjetoIndustrial, Caja, Cilindro
from src.horno import Horno
from src.robot_panda import RobotPanda

# Importación de los módulos procedurales base de la raíz
import planta_4_cintas
import robot as base_robot


class PlantaIndustrial:
    """
    Clase orquestadora (Composition Root) que encapsula la planta industrial completa.
    Gestiona la carga del modelo MuJoCo, la instanciación de cintas, cámaras, horno,
    robot y objetos, así como el bucle de simulación y renderizado.
    """
    def __init__(self, xml_path: str | None = "planta_industrial.xml"):
        # Carga o generación dinámica del modelo MuJoCo
        if xml_path and os.path.exists(xml_path) and os.path.getsize(xml_path) > 0:
            self.model = mujoco.MjModel.from_xml_path(xml_path)
        else:
            self.model = base_robot.build_model_with_franka(
                planta_4_cintas.build_mjcf(),
                planta_4_cintas.ROBOT_CONFIG
            )

        self.data = mujoco.MjData(self.model)

        # 1. Cintas Transportadoras
        self.cintas = [
            CintaTransportadora(1, "Cinta 1 · Inferior", velocidad=0.55),
            CintaTransportadora(2, "Cinta 2 · Derecha", velocidad=0.55),
            CintaTransportadora(3, "Cinta 3 · Superior", velocidad=0.55),
            CintaTransportadora(4, "Cinta 4 · Izquierda", velocidad=0.55),
            CintaTransportadora(5, "Cinta 5 · Horno", velocidad=0.48),
        ]

        # 2. Cámaras de Inspección (asociadas a Cinta 1 y Cinta 2)
        self.camara_bottom = CamaraInspeccion("cam_bottom", cinta=self.cintas[0], area_minima=300)
        self.camara_right = CamaraInspeccion("cam_right", cinta=self.cintas[1], area_minima=300)

        # 3. Objetos Industriales
        body_id_0 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obj0")
        body_id_1 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obj1")
        body_id_2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obj2")
        body_id_3 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obj3")
        body_id_4 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obj4")

        self.objetos: list[ObjetoIndustrial] = [
            Caja(id_objeto=0, nombre_body="obj0", body_id=body_id_0, segmento=3),
            Cilindro(id_objeto=1, nombre_body="obj1", body_id=body_id_1, segmento=1),
            Caja(id_objeto=2, nombre_body="obj2", body_id=body_id_2, segmento=2),
            Cilindro(id_objeto=3, nombre_body="obj3", body_id=body_id_3, segmento=3),
            Caja(id_objeto=4, nombre_body="obj4", body_id=body_id_4, segmento=0),
        ]

        #estructuras auxiliares para el movimiento físico sobre las cintas
        self.body_ids, self.dof_addrs = planta_4_cintas.get_ids(self.model)
        self.segments = [obj.segmento for obj in self.objetos]
        self.oven_counts = [0, 0, 0, 0, 0]
        planta_4_cintas.reset_objects(self.model, self.data, self.segments)

        # 4. Horno Industrial
        geom_id_horno = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "oven_hot_mat")
        if geom_id_horno < 0:
            geom_id_horno = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "oven_geom")

        self.horno = Horno(geom_id=geom_id_horno, temperatura_inicial=20.0)

        # 5. Robot Franka Panda
        self.robot_state = base_robot.new_robot_state()
        try:
            robot_ids = base_robot.get_robot_ids(self.model)
            base_robot.initialize_robot(self.model, self.data, robot_ids, self.robot_state)
            self.robot = RobotPanda(self.model, self.data, robot_ids)
            if self.robot_state.get("home_xmat") is not None:
                self.robot.orientacion_objetivo = self.robot_state["home_xmat"].copy()
        except Exception:
            self.robot = None

        self.renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._mostrar_opencv = True

    def actualizar_rodillos_cintas(self) -> None:
        """Sincroniza los actuadores de rodillos y banderas con el estado de cada cinta."""
        f1 = self.cintas[0].esta_activa()
        f2 = self.cintas[1].esta_activa()
        f3 = self.cintas[2].esta_activa()
        f4 = self.cintas[3].esta_activa()
        f5 = self.cintas[4].esta_activa()

        # Sincronizar banderas del módulo base para el transporte de fuerzas
        planta_4_cintas.BANDERA_CINTA_1 = f1
        planta_4_cintas.BANDERA_CINTA_2 = f2
        planta_4_cintas.BANDERA_CINTA_3 = f3
        planta_4_cintas.BANDERA_CINTA_4 = f4
        planta_4_cintas.BANDERA_CINTA_5 = f5

        comandos_rodillos = {
            "belt_act_b1": 5.5 if f1 else 0.0,
            "belt_act_b2": 5.5 if f1 else 0.0,
            "belt_act_r1": -5.5 if f2 else 0.0,
            "belt_act_r2": -5.5 if f2 else 0.0,
            "belt_act_t1": -5.5 if f3 else 0.0,
            "belt_act_t2": -5.5 if f3 else 0.0,
            "belt_act_l1": 5.5 if f4 else 0.0,
            "belt_act_l2": 5.5 if f4 else 0.0,
            "belt_act_o1": -5.5 if f5 else 0.0,
            "belt_act_o2": -5.5 if f5 else 0.0,
        }

        for nombre_actuador, valor in comandos_rodillos.items():
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, nombre_actuador)
            if aid >= 0:
                self.data.ctrl[aid] = valor

    def procesar_teclado(self, keycode: int) -> None:
        """Maneja los eventos de teclado para conmutar cintas y ajustar la temperatura del horno."""
        if 49 <= keycode <= 53:
            idx = keycode - 49
            self.cintas[idx].conmutar()
            estado = "ON" if self.cintas[idx].esta_activa() else "OFF"
            print(f"[Teclado] {self.cintas[idx].nombre} -> {estado}")
        elif keycode in (43, 61, 334):  # '+' o '=' o Numpad '+'
            self.horno.aumentar_temperatura(10.0)
            print(f"[Horno] Temperatura aumentada: {self.horno.temperatura:.1f} °C")
        elif keycode in (45, 333):      # '-' o Numpad '-'
            self.horno.disminuir_temperatura(10.0)
            print(f"[Horno] Temperatura disminuida: {self.horno.temperatura:.1f} °C")

    def actualizar(self) -> tuple[np.ndarray, np.ndarray]:
        """Paso de actualización principal."""
        self.actualizar_rodillos_cintas()

        # Aplicar fuerzas de las cintas transportadoras sobre las piezas
        planta_4_cintas.apply_transport(
            self.model, self.data, self.body_ids, self.dof_addrs,
            self.segments, self.oven_counts, self.robot_state
        )

        if self.robot:
            self.robot.actualizar()

        mujoco.mj_step(self.model, self.data)
        planta_4_cintas.lock_object_rotations(self.model, self.data)
        planta_4_cintas.lock_cylinders_to_belt_center(self.model, self.data, self.segments)
        mujoco.mj_forward(self.model, self.data)

        for i, obj in enumerate(self.objetos):
            obj.segmento = self.segments[i]
            obj.actualizar_posicion(self.data)

        frame_bottom = self.camara_bottom.actualizar(self.renderer, self.data)
        frame_right = self.camara_right.actualizar(self.renderer, self.data)

        self.horno.actualizar_color(self.model)

        return frame_bottom, frame_right

    def ejecutar(self) -> None:
        """Inicia el bucle principal de simulación."""
        with mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=self.procesar_teclado
        ) as viewer:
            while viewer.is_running():
                step_start = time.perf_counter()
                frame_b, frame_r = self.actualizar()
                mosaico = np.hstack((frame_b, frame_r))

                if self._mostrar_opencv:
                    try:
                        cv2.imshow("Camaras de Inspeccion (Cinta Inferior / Cinta Derecha)", mosaico)
                        key = cv2.waitKey(1) & 0xFF
                        if key != 255:
                            self.procesar_teclado(key)
                    except cv2.error:
                        # En macOS con mjpython, Cocoa bloquea cv2.imshow fuera del hilo principal
                        self._mostrar_opencv = False
                        print(
                            "[Aviso macOS] Ventana OpenCV deshabilitada por restricción de hilos de mjpython. "
                            "Usa las teclas 1-5, + y - directamente en la ventana 3D de MuJoCo."
                        )

                viewer.sync()

                elapsed = time.perf_counter() - step_start
                sleep_time = self.model.opt.timestep - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        if self._mostrar_opencv:
            cv2.destroyAllWindows()