"""
camera.py
=========
Thread webcam pour l'intégration Django du Body Language Model.

Capture la webcam, analyse chaque frame avec MediaPipe + SimpleBodyLanguageModel,
et expose :
  - Un flux MJPEG pour l'affichage live dans <img>
  - Les derniers scores (posture, confiance, stress) via get_latest_scores()

Usage dans views.py :
    from .camera import CameraStream
    stream = CameraStream.get_instance()
    stream.start()
"""

import threading
import time
import sys
import logging
import numpy as np
from pathlib import Path

log = logging.getLogger(__name__)

# ── Résultat par défaut quand la caméra n'est pas active ─────────────
DEFAULT_SCORES = {
    'posture'    : 'neutral',
    'confidence' : 0.0,
    'stress'     : 0.0,
    'arms_crossed': False,
    'face_touch'  : False,
    'active'      : False,
    'fill_ratio'  : 0.0,
    'coaching'    : 'Caméra non active',
}


class CameraStream:
    """
    Thread singleton qui gère la webcam et l'analyse Body Language.
    Utilise SimpleBodyLanguageModel (body_language_wrapper.py) — pas de pkl requis.
    """

    _instance = None
    _lock     = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._thread       = None
        self._running      = False
        self._frame_lock   = threading.Lock()
        self._scores_lock  = threading.Lock()
        self._latest_frame = None   # JPEG bytes
        self._latest_scores = DEFAULT_SCORES.copy()
        self._buffer       = None   # SequenceBuffer
        self._model        = None   # SimpleBodyLanguageModel

    # ── Démarrage ─────────────────────────────────────────────────────

    def start(self):
        """Démarre le thread webcam si pas déjà actif."""
        if self._running:
            return
        if not self._load_model():
            log.error("Impossible de charger le modèle Body Language.")
            return
        self._running = True
        self._thread  = threading.Thread(target=self._capture_loop,
                                         daemon=True)
        self._thread.start()
        log.info("CameraStream démarré.")

    def stop(self):
        """Arrête le thread webcam."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        self._latest_scores = DEFAULT_SCORES.copy()
        log.info("CameraStream arrêté.")

    @property
    def is_running(self):
        return self._running

    # ── Chargement du modèle ──────────────────────────────────────────

    def _load_model(self) -> bool:
        """Charge SimpleBodyLanguageModel et SequenceBuffer."""
        try:
            from django.conf import settings
            models_dir = str(settings.MODELS_AI_DIR)
            if models_dir not in sys.path:
                sys.path.insert(0, models_dir)

            from body_language_wrapper import SimpleBodyLanguageModel
            from lstm_model_v2         import SequenceBuffer

            self._model  = SimpleBodyLanguageModel(scaler=None)
            self._buffer = SequenceBuffer(seq_len=30)
            log.info("SimpleBodyLanguageModel chargé.")
            return True
        except Exception as e:
            log.error(f"Erreur chargement modèle : {e}")
            return False

    # ── Boucle de capture ─────────────────────────────────────────────

    def _capture_loop(self):
        """Boucle principale : capture webcam + MediaPipe + analyse."""
        try:
            import cv2
            import mediapipe as mp
            from mediapipe.tasks        import python
            from mediapipe.tasks.python import vision
            from django.conf            import settings

            # Initialiser MediaPipe
            model_path = Path(settings.MODELS_AI_DIR) / 'pose_landmarker.task'
            if not model_path.exists():
                # Fallback : chercher dans le dossier courant
                model_path = Path('pose_landmarker.task')

            if model_path.exists():
                base_options = python.BaseOptions(
                    model_asset_path=str(model_path))
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.VIDEO,
                    num_poses=1,
                    min_pose_detection_confidence=0.6,
                    min_pose_presence_confidence=0.6,
                    min_tracking_confidence=0.5,
                )
                detector  = vision.PoseLandmarker.create_from_options(options)
                use_mediapipe = True
                log.info("MediaPipe Pose Landmarker initialisé.")
            else:
                detector      = None
                use_mediapipe = False
                log.warning("pose_landmarker.task introuvable — mode dégradé (pas de landmarks).")

            cap       = cv2.VideoCapture(0)
            timestamp = 0

            if not cap.isOpened():
                log.error("Impossible d'ouvrir la webcam (index 0).")
                self._running = False
                return

            log.info("Webcam ouverte. Stream en cours...")

            while self._running:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                frame     = cv2.flip(frame, 1)
                timestamp += 33

                scores = DEFAULT_SCORES.copy()
                scores['active'] = True

                if use_mediapipe and detector:
                    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(
                        image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = detector.detect_for_video(mp_img, timestamp)

                    if result.pose_landmarks:
                        landmarks = result.pose_landmarks[0]
                        self._buffer.add_frame(landmarks)
                        scores['fill_ratio'] = self._buffer.fill_ratio

                        if self._buffer.is_ready():
                            cues = self._model.predict_from_buffer(
                                self._buffer, niveau_alia='Junior')
                            scores.update({
                                'posture'    : cues['posture_label'],
                                'confidence' : cues['confidence'],
                                'stress'     : cues['stress'],
                                'arms_crossed': cues['arms_crossed'],
                                'face_touch' : cues['face_touch'],
                                'coaching'   : self._get_coaching(
                                    cues['posture_label']),
                            })

                        # Dessiner skeleton
                        self._draw_skeleton(frame, landmarks)

                    # HUD overlay
                    self._draw_hud(frame, scores)

                else:
                    # Mode dégradé — afficher un message
                    cv2.putText(
                        frame,
                        'pose_landmarker.task manquant',
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Encoder en JPEG
                _, jpeg = cv2.imencode('.jpg', frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 70])
                with self._frame_lock:
                    self._latest_frame = jpeg.tobytes()

                with self._scores_lock:
                    self._latest_scores = scores

                time.sleep(0.033)  # ~30 fps

            cap.release()
            if use_mediapipe and detector:
                detector.close()

        except Exception as e:
            log.error(f"Erreur dans la boucle webcam : {e}")
            self._running = False

    # ── Getters ───────────────────────────────────────────────────────

    def get_latest_frame(self) -> bytes:
        """Retourne le dernier frame JPEG encodé."""
        with self._frame_lock:
            return self._latest_frame

    def get_latest_scores(self) -> dict:
        """Retourne les derniers scores analysés."""
        with self._scores_lock:
            return self._latest_scores.copy()

    # ── Helpers visuels ───────────────────────────────────────────────

    def _draw_skeleton(self, frame, landmarks):
        """Dessine le squelette MediaPipe sur le frame."""
        try:
            import cv2
            h, w = frame.shape[:2]
            CONNECTIONS = [
                (11,12),(11,13),(13,15),(12,14),(14,16),
                (11,23),(12,24),(23,24),
            ]
            pts = {i: (int(landmarks[i].x * w), int(landmarks[i].y * h))
                   for i in range(len(landmarks))}
            for a, b in CONNECTIONS:
                if a in pts and b in pts:
                    cv2.line(frame, pts[a], pts[b], (180, 60, 220), 2)
            for i in [0, 11, 12, 15, 16, 23, 24]:
                if i in pts:
                    cv2.circle(frame, pts[i], 5, (245, 117, 66), -1)
        except Exception:
            pass

    def _draw_hud(self, frame, scores):
        """Dessine le HUD avec confiance/stress/posture sur le frame."""
        try:
            import cv2

            # Fond semi-transparent
            overlay = frame.copy()
            cv2.rectangle(overlay, (8, 8), (300, 180), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

            # Barre Confidence
            conf = scores.get('confidence', 0)
            cv2.rectangle(frame, (18, 25), (278, 43), (50,50,50), -1)
            cv2.rectangle(frame, (18, 25),
                          (18 + int(260 * conf / 100), 43),
                          (0, 200, 80), -1)
            cv2.putText(frame, f"Confidence: {conf:.0f}%",
                        (18, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (230, 230, 230), 1)

            # Barre Stress
            stress = scores.get('stress', 0)
            cv2.rectangle(frame, (18, 65), (278, 83), (50,50,50), -1)
            cv2.rectangle(frame, (18, 65),
                          (18 + int(260 * stress / 100), 83),
                          (30, 30, 220), -1)
            cv2.putText(frame, f"Stress: {stress:.0f}%",
                        (18, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (230, 230, 230), 1)

            # Posture
            posture = scores.get('posture', 'neutral')
            colors  = {'upright':(0,220,0),'neutral':(0,200,255),'slouched':(0,60,255)}
            color   = colors.get(posture, (200, 200, 200))
            cv2.putText(frame, f"Posture: {posture}",
                        (18, 115), cv2.FONT_HERSHEY_SIMPLEX,
                        0.65, color, 2)

            # Arms / face
            arms  = scores.get('arms_crossed', False)
            face  = scores.get('face_touch', False)
            ac    = (30,30,220) if arms else (160,160,160)
            fc    = (30,30,220) if face else (160,160,160)
            cv2.putText(frame,
                        'Arms: CROSSED' if arms else 'Arms: open',
                        (18, 143), cv2.FONT_HERSHEY_SIMPLEX, 0.52, ac, 1)
            cv2.putText(frame,
                        'Hands: near face' if face else 'Hands: away',
                        (18, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.52, fc, 1)

            # Fill ratio si buffer pas plein
            fill = scores.get('fill_ratio', 1.0)
            if fill < 1.0:
                cv2.putText(frame,
                            f"Buffer: {fill*100:.0f}%",
                            (18, 185), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (200, 200, 100), 1)
        except Exception:
            pass

    def _get_coaching(self, posture: str) -> str:
        msgs = {
            'upright' : 'Excellente posture — maintenez cet engagement.',
            'neutral' : 'Posture correcte — ouvrez légèrement les épaules.',
            'slouched': '⚠️ Posture voûtée — redressez-vous.',
        }
        return msgs.get(posture, '')
