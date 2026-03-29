import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

import sys
sys.path.insert(0, r'C:\Users\MSI\Downloads\Projet DS\alia_django\models_ai')
from lstm_model_v2 import BodyLanguageModel, SequenceBuffer


def angle_between(v1, v2):
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def joint_angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    return angle_between(ba, bc)


# Rolling average buffer to smooth scores and reduce jitter
class Smoother:
    def __init__(self, window=10):
        self.window = window
        self.buf = {}

    def smooth(self, key, value):
        if key not in self.buf:
            self.buf[key] = []
        self.buf[key].append(value)
        if len(self.buf[key]) > self.window:
            self.buf[key].pop(0)
        return float(np.mean(self.buf[key]))

smoother = Smoother(window=12)


def analyze_upper_body(landmarks, frame_shape):
    h, w = frame_shape[:2]

    def lm(idx):
        p = landmarks[idx]
        return np.array([p.x, p.y])

    # Extract key landmark positions
    PL    = vision.PoseLandmark
    l_sh  = lm(PL.LEFT_SHOULDER.value)
    r_sh  = lm(PL.RIGHT_SHOULDER.value)
    l_hip = lm(PL.LEFT_HIP.value)
    r_hip = lm(PL.RIGHT_HIP.value)
    l_wr  = lm(PL.LEFT_WRIST.value)
    r_wr  = lm(PL.RIGHT_WRIST.value)
    l_el  = lm(PL.LEFT_ELBOW.value)
    r_el  = lm(PL.RIGHT_ELBOW.value)
    nose  = lm(PL.NOSE.value)

    mid_sh     = (l_sh  + r_sh)  / 2
    mid_hip    = (l_hip + r_hip) / 2
    shoulder_w = np.linalg.norm(l_sh - r_sh) + 1e-9

    # Compute slouch ratios: higher values indicate more slouching
    nose_to_sh_dist = abs(nose[1] - mid_sh[1])
    ratio_a         = shoulder_w / (nose_to_sh_dist + 1e-9)
    torso_height    = abs(mid_hip[1] - mid_sh[1])
    ratio_b         = shoulder_w / (torso_height + 1e-9)
    sh_tilt_raw     = abs(l_sh[1] - r_sh[1]) / shoulder_w

    slouch_a      = np.clip((ratio_a - 0.7) / 0.9, 0, 1)
    slouch_b      = np.clip((ratio_b - 0.5) / 0.7, 0, 1)
    slouch_t      = np.clip(sh_tilt_raw / 0.12, 0, 1)
    slouch_score  = slouch_a * 0.50 + slouch_b * 0.35 + slouch_t * 0.15
    slouch_smooth = smoother.smooth('slouch', slouch_score)

    # Classify posture: upright requires wide shoulders and hands at sides
    shoulders_wide   = ratio_a < 1.0
    l_wrist_near_hip = (abs(l_wr[1] - l_hip[1]) < shoulder_w * 0.8 and
                        abs(l_wr[0] - l_hip[0]) < shoulder_w * 1.0)
    r_wrist_near_hip = (abs(r_wr[1] - r_hip[1]) < shoulder_w * 0.8 and
                        abs(r_wr[0] - r_hip[0]) < shoulder_w * 1.0)
    hands_at_sides   = l_wrist_near_hip and r_wrist_near_hip
    is_upright       = shoulders_wide and hands_at_sides

    if is_upright:
        posture_label, posture_color = 'upright', (0, 220, 0)
    elif slouch_smooth < 0.45:
        posture_label, posture_color = 'neutral', (0, 200, 255)
    else:
        posture_label, posture_color = 'slouched', (0, 60, 255)

    spine_vec   = mid_hip - mid_sh
    spine_angle = angle_between(spine_vec, np.array([0.0, 1.0]))

    # Compute stress components
    sh_to_nose_dist  = abs(mid_sh[1] - nose[1])
    hunch_score      = max(0, 1 - sh_to_nose_dist / 0.20)

    mid_x        = mid_sh[0]
    buf          = shoulder_w * 0.05
    l_crossed    = l_wr[0] < (mid_x - buf)
    r_crossed    = r_wr[0] > (mid_x + buf)
    arms_crossed = l_crossed and r_crossed
    cross_score  = 1.0 if arms_crossed else ((int(l_crossed) + int(r_crossed)) * 0.4)

    wrist_dist  = np.linalg.norm(l_wr - r_wr) / shoulder_w
    wrist_score = max(0, 1 - wrist_dist / 0.6)

    l_near_face      = l_wr[1] < mid_sh[1] and abs(l_wr[0] - nose[0]) < shoulder_w * 0.6
    r_near_face      = r_wr[1] < mid_sh[1] and abs(r_wr[0] - nose[0]) < shoulder_w * 0.6
    face_touch_score = (int(l_near_face) + int(r_near_face)) / 2

    raw_stress = (hunch_score      * 0.22 +
                  cross_score      * 0.35 +
                  wrist_score      * 0.13 +
                  slouch_smooth    * 0.10 +
                  face_touch_score * 0.20) * 100

    # Stress is primary; confidence is its complement so they always sum to 100
    stress     = smoother.smooth('stress', np.clip(raw_stress, 0, 100))
    confidence = 100 - stress
    spine_sm   = smoother.smooth('spine', spine_angle)

    return {
        'confidence':    confidence,
        'stress':        stress,
        'posture_label': posture_label,
        'posture_color': posture_color,
        'spine_angle':   spine_sm,
        'slouch_score':  slouch_smooth,
        'arms_crossed':  arms_crossed,
        'face_touch':    l_near_face or r_near_face,
        'is_upright':    is_upright,
        'ratio_a':       ratio_a,
        'ratio_b':       ratio_b,
    }


# Draw pose skeleton on frame
POSE_CONNECTIONS = [
    (11,12),(11,13),(13,15),(12,14),(14,16),
    (11,23),(12,24),(23,24),
    (23,25),(25,27),(24,26),(26,28),
]

def draw_skeleton(frame, landmarks):
    h, w = frame.shape[:2]
    pts = {i: (int(landmarks[i].x*w), int(landmarks[i].y*h))
           for i in range(len(landmarks))}
    for a, b in POSE_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (180, 60, 220), 2)
    for pt in pts.values():
        cv2.circle(frame, pt, 4, (245, 117, 66), -1)


# Draw a labelled horizontal progress bar
def draw_bar(frame, x, y, w, h, value, color, label):
    cv2.rectangle(frame, (x, y), (x+w, y+h), (50,50,50), -1)
    cv2.rectangle(frame, (x, y), (x+int(w*value/100), y+h), color, -1)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (120,120,120), 1)
    cv2.putText(frame, f'{label}: {value:.0f}%', (x, y-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (230,230,230), 1)


# Draw the HUD overlay with all scores and indicators
def draw_hud(frame, cues):
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (320, 210), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    draw_bar(frame, 18, 30, 260, 18, cues['confidence'], (0,200,80),  'Confidence')
    draw_bar(frame, 18, 75, 260, 18, cues['stress'],     (30,30,220), 'Stress')

    cv2.putText(frame, f"Posture: {cues['posture_label']}", (18, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, cues['posture_color'], 2)

    cross_color = (30,30,220) if cues['arms_crossed'] else (160,160,160)
    cv2.putText(frame, 'Arms: CROSSED' if cues['arms_crossed'] else 'Arms: open',
                (18, 153), cv2.FONT_HERSHEY_SIMPLEX, 0.58, cross_color, 1)

    face_color = (30,30,220) if cues['face_touch'] else (160,160,160)
    cv2.putText(frame, 'Hands: near face' if cues['face_touch'] else 'Hands: away from face',
                (18, 178), cv2.FONT_HERSHEY_SIMPLEX, 0.58, face_color, 1)


# Initialise MediaPipe pose landmarker in video mode
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.6,
    min_pose_presence_confidence=0.6,
    min_tracking_confidence=0.5,
)
detector = vision.PoseLandmarker.create_from_options(options)

import lstm_model_v2
import sys
import pickle
import numpy as np

# Enregistre toutes les classes de lstm_model_v2 dans __main__
for _name in dir(lstm_model_v2):
    _obj = getattr(lstm_model_v2, _name)
    if isinstance(_obj, type):
        sys.modules['__main__'].__dict__[_name] = _obj

# Patch unpickler pour rediriger __main__ vers lstm_model_v2
class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == '__main__':
            if hasattr(lstm_model_v2, name):
                return getattr(lstm_model_v2, name)
        return super().find_class(module, name)

# Monkey-patch joblib pour utiliser notre unpickler
import joblib.numpy_pickle as _jnp
_orig_unpickle = _jnp.NumpyUnpickler
class _PatchedUnpickler(_orig_unpickle):
    def find_class(self, module, name):
        if module == '__main__':
            if hasattr(lstm_model_v2, name):
                return getattr(lstm_model_v2, name)
        return super().find_class(module, name)
_jnp.NumpyUnpickler = _PatchedUnpickler

import lstm_model_v2 as _lv2
_preloaded = getattr(_lv2, '_PRELOADED_MODEL', None)
if _preloaded is None:
    raise RuntimeError("Modèle non préchargé — lance run_body_language.py")
lstm_model = _preloaded

buffer = SequenceBuffer(seq_len=30)

# Main capture and detection loop
cap       = cv2.VideoCapture(0)
timestamp = 0

print("Running — press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame      = cv2.flip(frame, 1)
    timestamp += 33

    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(mp_img, timestamp)

    if result.pose_landmarks:
        landmarks = result.pose_landmarks[0]
        buffer.add_frame(landmarks)
        if buffer.is_ready():
            cues = lstm_model.get_hud_data(
                lstm_model.predict_from_buffer(buffer, niveau_alia='Junior')
            )
        else:
            cues = analyze_upper_body(landmarks, frame.shape)
        draw_skeleton(frame, landmarks)
        draw_hud(frame, cues)
    else:
        cv2.putText(frame, 'No person detected', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('Body Language Detector — Q to quit', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
detector.close()