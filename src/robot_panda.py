import mujoco
import numpy as np

# Parámetros de control de Cinemática Inversa (DLS)
ROBOT_IK_DAMPING = 2.0e-3
ROBOT_POS_GAIN = 3.0
ROBOT_ORI_GAIN = 1.4
ROBOT_NULL_GAIN = 0.35
ROBOT_MAX_QVEL = 1.15
PANDA_HOME_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=float)


class RobotPanda:
    """
    Encapsula el brazo robótico Franka Emika Panda.
    Gestiona el control del gripper, la trayectoria por cinemática inversa (IK)
    y el seguimiento de objetivos tridimensionales en la simulación MuJoCo.
    """
    def __init__(self, model, data, robot_ids: dict):
        self.model = model
        self.data = data
        self.robot_ids = robot_ids

        # Estado interno del robot
        self.pose_objetivo = np.array([-1.45, 0.05, 0.62], dtype=float)
        self.orientacion_objetivo = np.eye(3, dtype=float)
        self.gripper_cerrado = False
        self.ik_alcanzada = False

    def mover_a(self, posicion_3d: np.ndarray, matriz_rotacion: np.ndarray | None = None) -> None:
        """Establece las coordenadas cartesianas objetivo (X, Y, Z) para el efector final."""
        self.pose_objetivo = np.asarray(posicion_3d, dtype=float).copy()
        if matriz_rotacion is not None:
            self.orientacion_objetivo = np.asarray(matriz_rotacion, dtype=float).copy()

    def accionar_gripper(self, cerrar: bool) -> None:
        """Abre o cierra las pinzas del robot."""
        self.gripper_cerrado = cerrar
        val = 12.0 if cerrar else 255.0
        if "finger_actuator_ids" in self.robot_ids:
            for aid in self.robot_ids["finger_actuator_ids"]:
                self.data.ctrl[aid] = val

    def actualizar_ik(self) -> tuple[float, float]:
        """
        Ejecuta un paso del solucionador de Cinemática Inversa DLS (Damped Least Squares)
        para posicionar y orientar el TCP en self.pose_objetivo.
        """
        if not self.robot_ids or "tcp_site_id" not in self.robot_ids:
            return 0.0, 0.0

        site_id = self.robot_ids["tcp_site_id"]
        cur_pos = self.data.site_xpos[site_id].copy()
        cur_R = self.data.site_xmat[site_id].reshape(3, 3).copy()

        # 1. Error cartesiano en posición y orientación
        pos_err = self.pose_objetivo - cur_pos
        ori_err = 0.5 * (
            np.cross(cur_R[:, 0], self.orientacion_objetivo[:, 0])
            + np.cross(cur_R[:, 1], self.orientacion_objetivo[:, 1])
            + np.cross(cur_R[:, 2], self.orientacion_objetivo[:, 2])
        )

        # 2. Jacobiana de MuJoCo en el site del TCP
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)

        dofs = self.robot_ids.get("dof_addrs", [])
        if len(dofs) == 0:
            return 0.0, 0.0

        J = np.vstack((jacp[:, dofs], jacr[:, dofs]))

        # 3. Vector twist (velocidades cartesianas deseadas)
        twist = np.concatenate((ROBOT_POS_GAIN * pos_err, ROBOT_ORI_GAIN * ori_err))

        # 4. Solución por Mínimos Cuadrados Amortiguados (DLS)
        A = J @ J.T + ROBOT_IK_DAMPING * np.eye(6)
        try:
            y = np.linalg.solve(A, twist)
            J_pinv = J.T @ np.linalg.solve(A, np.eye(6))
        except np.linalg.LinAlgError:
            y, _, _, _ = np.linalg.lstsq(A, twist, rcond=None)
            J_pinv = np.linalg.pinv(J)

        dq = J.T @ y

        # 5. Corrección en el espacio nulo hacia la postura HOME
        q = self.data.qpos[self.robot_ids["qpos_addrs"]].copy()
        null = np.eye(len(dofs)) - J_pinv @ J
        dq += null @ (ROBOT_NULL_GAIN * (PANDA_HOME_Q[:len(dofs)] - q))
        dq = np.clip(dq, -ROBOT_MAX_QVEL, ROBOT_MAX_QVEL)

        # 6. Integración y límites de articulación
        q_cmd = q + dq * float(self.model.opt.timestep)
        for k, jid in enumerate(self.robot_ids.get("joint_ids", [])):
            if self.model.jnt_limited[jid]:
                lo, hi = self.model.jnt_range[jid]
                q_cmd[k] = np.clip(q_cmd[k], lo, hi)

        # Aplicar señal de control a los actuadores del robot
        if "actuator_ids" in self.robot_ids:
            self.data.ctrl[self.robot_ids["actuator_ids"]] = q_cmd

        err_pos_norm = float(np.linalg.norm(pos_err))
        err_ori_norm = float(np.linalg.norm(ori_err))
        self.ik_alcanzada = (err_pos_norm <= 0.035 and err_ori_norm <= np.deg2rad(4.0))

        return err_pos_norm, err_ori_norm

    def actualizar(self) -> None:
        """Paso de actualización periódica para recalcular la IK y mover el robot."""
        self.actualizar_ik()