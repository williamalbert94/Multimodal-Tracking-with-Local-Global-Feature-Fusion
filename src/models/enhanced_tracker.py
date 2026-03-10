"""
Enhanced Multi-Object Tracker con Motion-Based Matching
========================================================

Sistema de tracking mejorado que usa:
- Re-ID embeddings (appearance)
- Motion features (dirección, velocidad, distancia)
- Geometric features (tamaño, IoU)
- Memoria temporal (mantener tracks perdidos por max 10 frames)

Flujo:
1. Frame 0: Inicializar tracks con IDs únicos
2. Frame t>0: Asociar nuevos boxes con tracks existentes usando multi-cue matching
3. Tracks sin match: mantener en memoria por max_age frames antes de eliminar
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
import logging

logger = logging.getLogger(__name__)


class Track:
    """
    Representa un track individual con historial de movimiento.
    """
    # Contador global de IDs
    _next_id = 1000

    def __init__(self, box, embedding=None, track_id=None):
        """
        Inicializa un nuevo track.

        Args:
            box: [7] array - Bounding box [x, y, z, l, w, h, yaw]
            embedding: [D] array - Re-ID embedding (opcional)
            track_id: int - ID específico (opcional, auto-genera si None)
        """
        # ID único del track
        if track_id is not None:
            self.id = track_id
        else:
            self.id = Track._next_id
            Track._next_id += 1

        # Estado actual
        self.box = box  # [7] - último box observado
        self.embedding = embedding  # [D] - último embedding

        # Historial de movimiento
        self.positions = [box[:3].copy()]  # Lista de centroides [[x,y,z], ...]
        self.velocities = []  # Lista de velocidades [[vx,vy,vz], ...]
        self.boxes = [box.copy()]  # Historial de boxes

        # Embedding history (para promediar embeddings y mejorar robustez)
        self.embedding_history = []
        if embedding is not None:
            self.embedding_history.append(embedding.copy())

        # Gestión del track
        self.age = 0  # Frames desde creación
        self.hits = 1  # Número de veces detectado
        self.frames_since_update = 0  # Frames desde última actualización
        self.confirmed = False  # True si track está confirmado (min_hits alcanzado)

        # Motion features
        self.last_velocity = np.zeros(3)  # [vx, vy, vz]
        self.last_speed = 0.0
        self.last_direction = 0.0  # Ángulo en BEV

    def update(self, box, embedding=None):
        """
        Actualiza el track con nueva observación.

        Args:
            box: [7] - Nuevo bounding box
            embedding: [D] - Nuevo embedding (opcional)
        """
        # Calcular movimiento
        old_center = self.box[:3]
        new_center = box[:3]
        displacement = new_center - old_center

        # Actualizar velocidad (asumiendo dt=0.1s)
        dt = 0.1
        velocity = displacement / dt
        self.last_velocity = velocity
        self.last_speed = np.linalg.norm(velocity)

        # Dirección en BEV
        dx, dy = displacement[0], displacement[1]
        self.last_direction = np.arctan2(dy, dx)

        # Actualizar estado
        self.box = box
        self.positions.append(new_center.copy())
        self.velocities.append(velocity.copy())
        self.boxes.append(box.copy())

        # Actualizar embedding (EMA - Exponential Moving Average)
        if embedding is not None:
            if self.embedding is None:
                self.embedding = embedding.copy()
            else:
                alpha = 0.9  # Peso para embedding anterior
                self.embedding = alpha * self.embedding + (1 - alpha) * embedding

            self.embedding_history.append(embedding.copy())
            # Mantener solo últimos 10 embeddings
            if len(self.embedding_history) > 10:
                self.embedding_history = self.embedding_history[-10:]

        # Actualizar contadores
        self.hits += 1
        self.frames_since_update = 0
        self.age += 1

        # Limitar historial (últimos 20 frames)
        if len(self.positions) > 20:
            self.positions = self.positions[-20:]
            self.velocities = self.velocities[-20:]
            self.boxes = self.boxes[-20:]

    def predict(self):
        """
        Predice la posición del track en el siguiente frame.

        Returns:
            predicted_box: [7] - Box predicho
        """
        # Predicción simple: usar última velocidad
        dt = 0.1
        predicted_center = self.box[:3] + self.last_velocity * dt

        # Box predicho (mantener tamaño y orientación)
        predicted_box = self.box.copy()
        predicted_box[:3] = predicted_center

        return predicted_box

    def mark_missed(self):
        """Marca el track como no detectado en este frame."""
        self.frames_since_update += 1
        self.age += 1

    def is_confirmed(self, min_hits=2):
        """Verifica si el track está confirmado."""
        return self.hits >= min_hits

    def should_delete(self, max_age=10):
        """Verifica si el track debe ser eliminado."""
        return self.frames_since_update >= max_age


class EnhancedTracker:
    """
    Tracker mejorado con motion-based matching y memoria temporal.

    Características:
    - Multi-cue matching (appearance + motion + geometry)
    - Memoria temporal de tracks perdidos (max_age)
    - Confirmación de tracks (min_hits)
    - Gestión automática de IDs
    """

    def __init__(self, max_age=10, min_hits=1, matching_threshold=0.3):
        """
        Args:
            max_age: int - Máximo de frames sin detección antes de eliminar track
            min_hits: int - Mínimo de detecciones para confirmar track
            matching_threshold: float - Threshold para aceptar asociación
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.matching_threshold = matching_threshold

        self.tracks = []  # Lista de tracks activos
        self.frame_count = 0

        logger.info(f"EnhancedTracker initialized:")
        logger.info(f"  max_age={max_age} (memoria temporal)")
        logger.info(f"  min_hits={min_hits} (confirmación)")
        logger.info(f"  matching_threshold={matching_threshold}")

    def update(self, boxes, embeddings=None, motion_dict=None):
        """
        Actualiza tracker con nuevas detecciones.

        Args:
            boxes: [M, 7] - Detected boxes
            embeddings: [M, D] - Re-ID embeddings (opcional)
            motion_dict: Dict - Motion features calculadas por motion_utils

        Returns:
            output_tracks: List[(box, track_id)] - Tracks confirmados
        """
        self.frame_count += 1

        # Convertir a numpy
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        M = len(boxes)  # Número de detecciones
        N = len(self.tracks)  # Número de tracks existentes

        # ===== CASO 1: Primer frame (inicialización) =====
        if self.frame_count == 1:
            logger.info(f"Frame 0: Initializing {M} tracks")
            for i in range(M):
                emb = embeddings[i] if embeddings is not None else None
                track = Track(boxes[i], emb)
                self.tracks.append(track)
                logger.debug(f"  Created track {track.id}")

            # Retornar todos los tracks iniciales
            return [(t.box, t.id) for t in self.tracks]

        # ===== CASO 2: Frames subsecuentes (matching) =====

        # Predecir posiciones de tracks existentes
        predicted_boxes = []
        for track in self.tracks:
            pred_box = track.predict()
            predicted_boxes.append(pred_box)

        # Construir matriz de costos para Hungarian matching
        cost_matrix = self._build_cost_matrix(
            self.tracks, boxes, embeddings, motion_dict
        )

        # Resolver asignación óptima (Hungarian algorithm)
        if N > 0 and M > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            matched_indices = list(zip(row_indices, col_indices))
        else:
            matched_indices = []

        # Filtrar matches con costo muy alto (threshold)
        valid_matches = []
        for row, col in matched_indices:
            if cost_matrix[row, col] < (1.0 - self.matching_threshold):
                valid_matches.append((row, col))

        matched_track_indices = set([m[0] for m in valid_matches])
        matched_det_indices = set([m[1] for m in valid_matches])

        logger.debug(f"Frame {self.frame_count}: {len(valid_matches)}/{M} detections matched")

        # ===== ACTUALIZAR TRACKS MATCHED =====
        for track_idx, det_idx in valid_matches:
            emb = embeddings[det_idx] if embeddings is not None else None
            self.tracks[track_idx].update(boxes[det_idx], emb)

        # ===== MARCAR TRACKS NO MATCHED COMO "MISSED" =====
        for track_idx in range(N):
            if track_idx not in matched_track_indices:
                self.tracks[track_idx].mark_missed()

        # ===== CREAR NUEVOS TRACKS PARA DETECCIONES SIN MATCH =====
        new_track_count = 0
        for det_idx in range(M):
            if det_idx not in matched_det_indices:
                emb = embeddings[det_idx] if embeddings is not None else None
                track = Track(boxes[det_idx], emb)
                self.tracks.append(track)
                new_track_count += 1
                logger.debug(f"  New track {track.id} (unmatched detection)")

        if new_track_count > 0:
            logger.debug(f"  Created {new_track_count} new tracks")

        # ===== ELIMINAR TRACKS PERDIDOS (>max_age frames sin detección) =====
        deleted_count = 0
        self.tracks = [t for t in self.tracks if not t.should_delete(self.max_age)]
        if deleted_count > 0:
            logger.debug(f"  Deleted {deleted_count} lost tracks (>max_age)")

        # ===== RETORNAR TRACKS CONFIRMADOS =====
        output_tracks = []
        for track in self.tracks:
            # Solo retornar tracks confirmados y recientemente actualizados
            if track.is_confirmed(self.min_hits) and track.frames_since_update == 0:
                output_tracks.append((track.box, track.id))

        logger.debug(f"  Output: {len(output_tracks)} confirmed tracks")

        return output_tracks

    def _build_cost_matrix(self, tracks, boxes, embeddings, motion_dict):
        """
        Construye matriz de costos para matching usando multi-cue.

        Costos combinados:
        - Appearance (Re-ID similarity): weight=0.4
        - Motion (movement consistency): weight=0.3
        - IoU (geometric overlap): weight=0.2
        - Size (box size similarity): weight=0.1

        Args:
            tracks: List[Track]
            boxes: [M, 7]
            embeddings: [M, D]
            motion_dict: Dict

        Returns:
            cost_matrix: [N, M] - Menores costos = mejores matches
        """
        N = len(tracks)
        M = len(boxes)

        if N == 0 or M == 0:
            return np.zeros((N, M))

        cost_matrix = np.ones((N, M))  # Inicializar con alto costo

        for i, track in enumerate(tracks):
            for j in range(M):
                # ===== APPEARANCE SIMILARITY =====
                appearance_score = 0.5
                if track.embedding is not None and embeddings is not None:
                    emb1 = track.embedding / (np.linalg.norm(track.embedding) + 1e-8)
                    emb2 = embeddings[j] / (np.linalg.norm(embeddings[j]) + 1e-8)
                    cosine_sim = np.dot(emb1, emb2)
                    appearance_score = (cosine_sim + 1) / 2  # Normalizar a [0,1]

                # ===== MOTION CONSISTENCY =====
                motion_score = 0.5
                if motion_dict is not None and track.id in motion_dict:
                    motion = motion_dict[track.id]
                    # Predecir posición esperada
                    predicted_center = motion['center_t1']
                    observed_center = boxes[j][:3]
                    error = np.linalg.norm(predicted_center - observed_center)
                    # Score alto si error bajo (sigma=1.0m)
                    motion_score = np.exp(-0.5 * (error / 1.0) ** 2)

                # ===== GEOMETRIC IoU =====
                iou = self._compute_iou_3d(track.box, boxes[j])

                # ===== SIZE SIMILARITY =====
                size1 = track.box[3:6]
                size2 = boxes[j][3:6]
                size_diff = np.linalg.norm(size1 - size2)
                size_score = np.exp(-size_diff)

                # ===== COMBINAR EN SCORE TOTAL =====
                total_score = (
                    0.4 * appearance_score +
                    0.3 * motion_score +
                    0.2 * iou +
                    0.1 * size_score
                )

                # Convertir score [0,1] a costo (invertir: alto score = bajo costo)
                cost_matrix[i, j] = 1.0 - total_score

        return cost_matrix

    def _compute_iou_3d(self, box1, box2):
        """IoU simplificado (axis-aligned)."""
        c1, s1 = box1[:3], box1[3:6]
        c2, s2 = box2[:3], box2[3:6]

        min1 = c1 - s1 / 2
        max1 = c1 + s1 / 2
        min2 = c2 - s2 / 2
        max2 = c2 + s2 / 2

        inter_min = np.maximum(min1, min2)
        inter_max = np.minimum(max1, max2)
        inter_size = np.maximum(0, inter_max - inter_min)
        inter_volume = np.prod(inter_size)

        volume1 = np.prod(s1)
        volume2 = np.prod(s2)
        union_volume = volume1 + volume2 - inter_volume

        return inter_volume / union_volume if union_volume > 0 else 0.0

    def reset(self):
        """Reset tracker para nueva secuencia."""
        self.tracks = []
        self.frame_count = 0
        Track._next_id = 1000  # Reset ID counter
        logger.info("Tracker reset for new sequence")

    def get_active_tracks(self):
        """Retorna tracks activos (incluyendo los en memoria)."""
        return [(t.box, t.id, t.frames_since_update) for t in self.tracks]
