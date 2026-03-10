"""
Gallery-Based Tracker with Spatial Configuration Awareness
===========================================================

Sistema de tracking avanzado que mantiene un "gallery/buffer" con:
- Embeddings Re-ID
- Geometría del box (tamaño, posición)
- Número de puntos 3D dentro del box
- Vector de dirección/movimiento
- Distancias pairwise entre objetos
- Orden espacial (izquierda→derecha, adelante→atrás)

Autor: Sistema de Tracking Espacial Avanzado
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
import logging

logger = logging.getLogger(__name__)


class TrackGallery:
    """
    Gallery/Buffer que mantiene información completa de un track.

    Almacena:
    - Embeddings Re-ID (historial)
    - Box geometry (posición, tamaño, orientación)
    - Puntos 3D (número de puntos dentro del box)
    - Motion (dirección, velocidad)
    - Spatial context (orden relativo, distancias a otros objetos)
    """

    # Contador global de IDs
    _next_id = 1000

    def __init__(self, box, embedding=None, num_points=0, track_id=None):
        """
        Args:
            box: [7] - Bounding box [x, y, z, l, w, h, yaw]
            embedding: [D] - Re-ID embedding
            num_points: int - Número de puntos 3D dentro del box
            track_id: int - ID específico (opcional)
        """
        # ID único
        if track_id is not None:
            self.id = track_id
        else:
            self.id = TrackGallery._next_id
            TrackGallery._next_id += 1

        # ===== APPEARANCE FEATURES =====
        self.embedding = embedding
        self.embedding_history = [embedding] if embedding is not None else []

        # ===== GEOMETRY FEATURES =====
        self.box = box  # [7] - último box observado
        self.box_history = [box.copy()]
        self.center = box[:3]  # [x, y, z]
        self.size = box[3:6]   # [l, w, h]
        self.yaw = box[6]

        # ===== DENSITY FEATURES =====
        self.num_points = num_points  # Número de puntos en el box
        self.num_points_history = [num_points]
        self.avg_num_points = num_points

        # ===== MOTION FEATURES =====
        self.velocity = np.zeros(3)  # [vx, vy, vz]
        self.direction_2d = 0.0      # Ángulo en BEV
        self.speed = 0.0             # Magnitud
        self.velocity_history = []

        # ===== SPATIAL CONTEXT =====
        # Distancias a otros objetos (dict: {other_track_id: distance})
        self.distances_to_others = {}
        # Orden relativo (ej: "left_of": [id1, id2], "right_of": [id3])
        self.spatial_order = {
            'left_of': [],   # IDs de tracks a la derecha
            'right_of': [],  # IDs de tracks a la izquierda
            'front_of': [],  # IDs de tracks atrás
            'behind': []     # IDs de tracks adelante
        }

        # ===== TRACK MANAGEMENT =====
        self.age = 0
        self.hits = 1
        self.frames_since_update = 0
        self.confirmed = False

    def update(self, box, embedding=None, num_points=0, spatial_context=None):
        """
        Actualiza el track con nueva observación.

        Args:
            box: [7] - Nuevo box
            embedding: [D] - Nuevo embedding
            num_points: int - Número de puntos en el nuevo box
            spatial_context: Dict - Contexto espacial (distancias, orden)
        """
        # Calcular movimiento
        old_center = self.center
        new_center = box[:3]
        displacement = new_center - old_center

        # Actualizar velocidad
        dt = 0.1  # Tiempo entre frames
        self.velocity = displacement / dt
        self.speed = np.linalg.norm(self.velocity)
        self.direction_2d = np.arctan2(displacement[1], displacement[0])

        # Actualizar geometría
        self.box = box
        self.center = new_center
        self.size = box[3:6]
        self.yaw = box[6]
        self.box_history.append(box.copy())

        # Actualizar embedding (EMA)
        if embedding is not None:
            if self.embedding is None:
                self.embedding = embedding
            else:
                alpha = 0.9
                self.embedding = alpha * self.embedding + (1 - alpha) * embedding
            self.embedding_history.append(embedding)
            if len(self.embedding_history) > 10:
                self.embedding_history = self.embedding_history[-10:]

        # Actualizar número de puntos (EMA)
        self.num_points = num_points
        self.num_points_history.append(num_points)
        if len(self.num_points_history) > 10:
            self.num_points_history = self.num_points_history[-10:]
        self.avg_num_points = np.mean(self.num_points_history)

        # Actualizar contexto espacial
        if spatial_context is not None:
            if 'distances' in spatial_context:
                self.distances_to_others = spatial_context['distances']
            if 'order' in spatial_context:
                self.spatial_order = spatial_context['order']

        # Actualizar motion history
        self.velocity_history.append(self.velocity.copy())
        if len(self.velocity_history) > 10:
            self.velocity_history = self.velocity_history[-10:]

        # Actualizar contadores
        self.hits += 1
        self.frames_since_update = 0
        self.age += 1

        # Limitar historial
        if len(self.box_history) > 20:
            self.box_history = self.box_history[-20:]

    def predict(self):
        """Predice posición en siguiente frame."""
        dt = 0.1
        predicted_center = self.center + self.velocity * dt
        predicted_box = self.box.copy()
        predicted_box[:3] = predicted_center
        return predicted_box

    def mark_missed(self):
        """Marca el track como no detectado."""
        self.frames_since_update += 1
        self.age += 1

    def is_confirmed(self, min_hits=2):
        """Verifica si está confirmado."""
        return self.hits >= min_hits

    def should_delete(self, max_age=10):
        """Verifica si debe eliminarse."""
        return self.frames_since_update >= max_age

    def get_feature_vector(self):
        """
        Retorna vector de features completo para matching.

        Returns:
            features: Dict con todas las características
        """
        return {
            'embedding': self.embedding,
            'center': self.center,
            'size': self.size,
            'num_points': self.avg_num_points,
            'velocity': self.velocity,
            'direction': self.direction_2d,
            'speed': self.speed,
            'distances_to_others': self.distances_to_others,
            'spatial_order': self.spatial_order,
        }


def compute_spatial_configuration(boxes, track_ids):
    """
    Calcula la configuración espacial de objetos en un frame.

    Args:
        boxes: [M, 7] - Boxes
        track_ids: [M] - Track IDs correspondientes

    Returns:
        spatial_config: Dict {track_id: {'distances': {...}, 'order': {...}}}
    """
    M = len(boxes)
    if M == 0:
        return {}

    # Extraer centroides
    centers = boxes[:, :3]  # [M, 3]

    spatial_config = {}

    for i, tid_i in enumerate(track_ids):
        # Calcular distancias a todos los demás objetos
        distances = {}
        left_of = []
        right_of = []
        front_of = []
        behind = []

        for j, tid_j in enumerate(track_ids):
            if i == j:
                continue

            # Distancia euclidiana 2D (BEV)
            dist = np.linalg.norm(centers[i, :2] - centers[j, :2])
            distances[int(tid_j)] = float(dist)

            # Orden espacial (left/right, front/back)
            dx = centers[j, 0] - centers[i, 0]  # Diferencia en X
            dy = centers[j, 1] - centers[i, 1]  # Diferencia en Y

            # Threshold para considerar "mismo eje"
            threshold = 0.5  # metros

            if abs(dy) < threshold:  # Mismo eje Y (adelante/atrás)
                if dx > 0:
                    front_of.append(int(tid_j))  # tid_j está adelante
                else:
                    behind.append(int(tid_j))    # tid_j está atrás

            if abs(dx) < threshold:  # Mismo eje X (izquierda/derecha)
                if dy > 0:
                    left_of.append(int(tid_j))   # tid_j está a la izquierda
                else:
                    right_of.append(int(tid_j))  # tid_j está a la derecha

        spatial_config[int(tid_i)] = {
            'distances': distances,
            'order': {
                'left_of': left_of,
                'right_of': right_of,
                'front_of': front_of,
                'behind': behind,
            }
        }

    return spatial_config


def compute_spatial_consistency_score(track_order_prev, detection_order_curr, track_id, detection_idx, all_track_ids):
    """
    Calcula score de consistencia espacial entre frames.

    Verifica si el orden relativo de objetos se mantiene:
    - Si track_i estaba a la izquierda de track_j en t-1,
      ¿sigue estando a la izquierda en t?

    Returns:
        score: float [0, 1] - 1 = orden completamente consistente
    """
    if track_order_prev is None or detection_order_curr is None:
        return 0.5  # Neutral

    # Comparar órdenes relativos
    matches = 0
    total = 0

    for direction in ['left_of', 'right_of', 'front_of', 'behind']:
        prev_neighbors = track_order_prev.get(direction, [])
        curr_neighbors = detection_order_curr.get(direction, [])

        # Contar cuántos vecinos se mantienen en la misma dirección
        common = set(prev_neighbors) & set(all_track_ids)
        for neighbor_id in common:
            total += 1
            if neighbor_id in curr_neighbors:
                matches += 1

    if total == 0:
        return 0.5  # No hay vecinos para comparar

    return matches / total


# ===== MAIN GALLERY TRACKER =====
class GalleryTracker:
    """
    Gallery-Based Tracker con spatial configuration awareness.

    Usa TrackGallery para mantener información completa de cada track
    y realiza matching multi-cue considerando:
    - Embeddings (appearance)
    - Box geometry (size, position)
    - Density (num_points)
    - Motion (velocity, direction)
    - Spatial context (distances, ordering)
    """

    def __init__(self, max_age=10, min_hits=1, matching_threshold=0.3,
                 weight_appearance=0.30, weight_geometry=0.20, weight_density=0.10,
                 weight_motion=0.20, weight_spatial=0.20):
        """
        Args:
            max_age: Frames sin detección antes de eliminar track
            min_hits: Detecciones mínimas para confirmar track
            matching_threshold: Threshold de score para aceptar asociación
            weight_*: Pesos para cada componente del matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.matching_threshold = matching_threshold

        # Pesos para matching
        self.weight_appearance = weight_appearance
        self.weight_geometry = weight_geometry
        self.weight_density = weight_density
        self.weight_motion = weight_motion
        self.weight_spatial = weight_spatial

        # Gallery de tracks
        self.tracks = []  # List[TrackGallery]
        self.frame_count = 0

        logger.info("=" * 70)
        logger.info("GalleryTracker initialized")
        logger.info(f"  max_age={max_age}, min_hits={min_hits}")
        logger.info(f"  Weights: app={weight_appearance:.2f}, geom={weight_geometry:.2f}, "
                   f"dens={weight_density:.2f}, motion={weight_motion:.2f}, spatial={weight_spatial:.2f}")
        logger.info("=" * 70)

    def update(self, boxes, embeddings=None, num_points_per_box=None, motion_dict=None):
        """
        Actualiza tracker con nuevas detecciones.

        Args:
            boxes: [M, 7] - Detected boxes
            embeddings: [M, D] - Re-ID embeddings
            num_points_per_box: [M] - Número de puntos por box
            motion_dict: Dict - Motion features calculadas

        Returns:
            output_tracks: List[(box, track_id)] - Tracks confirmados
        """
        self.frame_count += 1

        # Convertir a numpy
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        M = len(boxes)
        N = len(self.tracks)

        logger.debug(f"\n{'=' * 60}")
        logger.debug(f"Frame {self.frame_count}: {M} detections, {N} existing tracks")

        # ===== CASO 1: PRIMER FRAME (Inicialización) =====
        if self.frame_count == 1:
            logger.info(f"Frame 0: Initializing {M} tracks")
            for i in range(M):
                emb = embeddings[i] if embeddings is not None else None
                npts = num_points_per_box[i] if num_points_per_box is not None else 0
                track = TrackGallery(boxes[i], emb, npts)
                self.tracks.append(track)
                logger.debug(f"  Created track {track.id}")

            # Calcular configuración espacial inicial
            if M > 1:
                track_ids = [t.id for t in self.tracks]
                spatial_config = compute_spatial_configuration(boxes, track_ids)
                for track in self.tracks:
                    if track.id in spatial_config:
                        track.distances_to_others = spatial_config[track.id]['distances']
                        track.spatial_order = spatial_config[track.id]['order']

            return [(t.box, t.id) for t in self.tracks]

        # ===== CASO 2: FRAMES SUBSECUENTES (Matching) =====

        # Calcular configuración espacial de detecciones actuales
        detection_track_ids_temp = list(range(M))  # IDs temporales
        spatial_config_detections = compute_spatial_configuration(boxes, detection_track_ids_temp)

        # Construir matriz de costos
        cost_matrix = self._build_cost_matrix(
            boxes, embeddings, num_points_per_box, motion_dict, spatial_config_detections
        )

        # Hungarian matching
        if N > 0 and M > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            matched_indices = list(zip(row_indices, col_indices))
        else:
            matched_indices = []

        # Filtrar matches con costo muy alto
        valid_matches = []
        for row, col in matched_indices:
            score = 1.0 - cost_matrix[row, col]
            if score >= self.matching_threshold:
                valid_matches.append((row, col))
                logger.debug(f"  Match: track {self.tracks[row].id} ↔ detection {col} (score={score:.3f})")

        matched_track_indices = set([m[0] for m in valid_matches])
        matched_det_indices = set([m[1] for m in valid_matches])

        logger.debug(f"  Valid matches: {len(valid_matches)}/{M}")

        # ===== ACTUALIZAR TRACKS MATCHED =====
        for track_idx, det_idx in valid_matches:
            emb = embeddings[det_idx] if embeddings is not None else None
            npts = num_points_per_box[det_idx] if num_points_per_box is not None else 0

            # Spatial context para este track
            spatial_ctx = None
            if det_idx in spatial_config_detections:
                # Mapear IDs temporales a IDs reales de tracks
                real_distances = {}
                real_order = {'left_of': [], 'right_of': [], 'front_of': [], 'behind': []}

                for temp_id, dist in spatial_config_detections[det_idx]['distances'].items():
                    # Encontrar track ID real correspondiente
                    if temp_id in matched_det_indices:
                        real_track_idx = [m[0] for m in valid_matches if m[1] == temp_id][0]
                        real_track_id = self.tracks[real_track_idx].id
                        real_distances[real_track_id] = dist

                # Similar para order
                for direction, temp_ids in spatial_config_detections[det_idx]['order'].items():
                    for temp_id in temp_ids:
                        if temp_id in matched_det_indices:
                            real_track_idx = [m[0] for m in valid_matches if m[1] == temp_id][0]
                            real_track_id = self.tracks[real_track_idx].id
                            real_order[direction].append(real_track_id)

                spatial_ctx = {'distances': real_distances, 'order': real_order}

            self.tracks[track_idx].update(boxes[det_idx], emb, npts, spatial_ctx)

        # ===== MARCAR TRACKS NO MATCHED =====
        for track_idx in range(N):
            if track_idx not in matched_track_indices:
                self.tracks[track_idx].mark_missed()

        # ===== CREAR NUEVOS TRACKS =====
        new_track_ids = []
        for det_idx in range(M):
            if det_idx not in matched_det_indices:
                emb = embeddings[det_idx] if embeddings is not None else None
                npts = num_points_per_box[det_idx] if num_points_per_box is not None else 0
                track = TrackGallery(boxes[det_idx], emb, npts)
                self.tracks.append(track)
                new_track_ids.append(track.id)
                logger.debug(f"  New track {track.id}")

        if len(new_track_ids) > 0:
            logger.debug(f"  Created {len(new_track_ids)} new tracks")

        # ===== ELIMINAR TRACKS PERDIDOS =====
        before_delete = len(self.tracks)
        self.tracks = [t for t in self.tracks if not t.should_delete(self.max_age)]
        deleted = before_delete - len(self.tracks)
        if deleted > 0:
            logger.debug(f"  Deleted {deleted} lost tracks")

        # ===== RETORNAR TRACKS CONFIRMADOS =====
        output_tracks = []
        for track in self.tracks:
            if track.is_confirmed(self.min_hits) and track.frames_since_update == 0:
                output_tracks.append((track.box, track.id))

        logger.debug(f"  Output: {len(output_tracks)} confirmed tracks")

        return output_tracks

    def _build_cost_matrix(self, boxes, embeddings, num_points_per_box, motion_dict, spatial_config_detections):
        """
        Construye matriz de costos para matching multi-cue.

        Componentes:
        1. Appearance: Cosine similarity de embeddings
        2. Geometry: IoU 3D + box size similarity
        3. Density: Similaridad en número de puntos
        4. Motion: Consistencia de dirección/velocidad
        5. Spatial: Consistencia de orden relativo y distancias

        Args:
            boxes: [M, 7]
            embeddings: [M, D]
            num_points_per_box: [M]
            motion_dict: Dict
            spatial_config_detections: Dict

        Returns:
            cost_matrix: [N, M] - Menor costo = mejor match
        """
        N = len(self.tracks)
        M = len(boxes)

        if N == 0 or M == 0:
            return np.zeros((N, M))

        cost_matrix = np.ones((N, M))  # Inicializar con alto costo

        for i, track in enumerate(self.tracks):
            for j in range(M):
                scores = {}

                # ===== 1. APPEARANCE SCORE =====
                appearance_score = 0.5  # Neutral por defecto
                if track.embedding is not None and embeddings is not None:
                    emb1 = track.embedding / (np.linalg.norm(track.embedding) + 1e-8)
                    emb2 = embeddings[j] / (np.linalg.norm(embeddings[j]) + 1e-8)
                    cosine_sim = np.dot(emb1, emb2)
                    appearance_score = (cosine_sim + 1) / 2  # [-1,1] → [0,1]
                scores['appearance'] = appearance_score

                # ===== 2. GEOMETRY SCORE =====
                # IoU 3D
                iou = self._compute_iou_3d(track.box, boxes[j])
                # Size similarity
                size_diff = np.linalg.norm(track.size - boxes[j][3:6])
                size_score = np.exp(-size_diff / 2.0)  # Gaussiano
                geometry_score = 0.7 * iou + 0.3 * size_score
                scores['geometry'] = geometry_score

                # ===== 3. DENSITY SCORE =====
                density_score = 0.5
                if num_points_per_box is not None and track.avg_num_points > 0:
                    ratio = min(num_points_per_box[j], track.avg_num_points) / max(num_points_per_box[j], track.avg_num_points)
                    density_score = ratio
                scores['density'] = density_score

                # ===== 4. MOTION SCORE =====
                motion_score = 0.5
                if motion_dict is not None and track.id in motion_dict:
                    motion = motion_dict[track.id]
                    # Predecir posición esperada
                    predicted_center = motion.get('center_t1', track.center)
                    observed_center = boxes[j][:3]
                    error = np.linalg.norm(predicted_center - observed_center)
                    # Gaussiano (sigma=1.5m)
                    motion_score = np.exp(-0.5 * (error / 1.5) ** 2)

                    # Consistencia de dirección
                    if track.speed > 0.5:  # Solo si se está moviendo
                        track_dir = track.direction_2d
                        motion_dir = motion.get('direction_2d', track_dir)
                        angle_diff = abs(track_dir - motion_dir)
                        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)  # Mínimo ángulo
                        direction_score = np.exp(-angle_diff)
                        motion_score = 0.7 * motion_score + 0.3 * direction_score

                scores['motion'] = motion_score

                # ===== 5. SPATIAL CONSISTENCY SCORE =====
                spatial_score = 0.5
                if len(track.distances_to_others) > 0 and j in spatial_config_detections:
                    # Comparar distancias a otros objetos
                    dist_consistency = []
                    for other_track_id, prev_dist in track.distances_to_others.items():
                        # Buscar este track en detecciones actuales
                        for temp_id, curr_dist in spatial_config_detections[j]['distances'].items():
                            # Verificar si temp_id corresponde a other_track_id
                            # (simplificado: usar distancias directamente)
                            dist_diff = abs(prev_dist - curr_dist)
                            consistency = np.exp(-dist_diff / 2.0)
                            dist_consistency.append(consistency)

                    if len(dist_consistency) > 0:
                        spatial_score = np.mean(dist_consistency)

                    # Comparar orden relativo
                    if j in spatial_config_detections:
                        order_score = compute_spatial_consistency_score(
                            track.spatial_order,
                            spatial_config_detections[j]['order'],
                            track.id,
                            j,
                            [t.id for t in self.tracks]
                        )
                        spatial_score = 0.6 * spatial_score + 0.4 * order_score

                scores['spatial'] = spatial_score

                # ===== COMBINAR SCORES =====
                total_score = (
                    self.weight_appearance * scores['appearance'] +
                    self.weight_geometry * scores['geometry'] +
                    self.weight_density * scores['density'] +
                    self.weight_motion * scores['motion'] +
                    self.weight_spatial * scores['spatial']
                )

                # Convertir a costo (invertir)
                cost_matrix[i, j] = 1.0 - total_score

        return cost_matrix

    def _compute_iou_3d(self, box1, box2):
        """Calcula IoU 3D simplificado (axis-aligned)."""
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
        """Reset para nueva secuencia."""
        self.tracks = []
        self.frame_count = 0
        TrackGallery._next_id = 1000
        logger.info("GalleryTracker reset")

    def get_active_tracks(self):
        """Retorna tracks activos (incluyendo en memoria)."""
        return [(t.box, t.id, t.frames_since_update) for t in self.tracks]


# ===== LOGGING =====
def log_gallery_statistics(tracks):
    """Log estadísticas del gallery para debugging."""
    if len(tracks) == 0:
        logger.info("Gallery: No tracks")
        return

    confirmed = sum([t.is_confirmed() for t in tracks])
    avg_points = np.mean([t.avg_num_points for t in tracks])
    avg_speed = np.mean([t.speed for t in tracks])

    logger.info(f"Gallery Statistics:")
    logger.info(f"  Total tracks: {len(tracks)}")
    logger.info(f"  Confirmed: {confirmed}/{len(tracks)}")
    logger.info(f"  Avg points/track: {avg_points:.1f}")
    logger.info(f"  Avg speed: {avg_speed:.2f} m/s")
