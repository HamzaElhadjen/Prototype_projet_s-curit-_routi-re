# main.py
# =============================================================================
#  INTERFACE PyQt5 "Cockpit conducteur" (style Audi / Volkswagen)
#  - Centre : vidéo (YOLO) avec anneau LED (bleu/jaune/rouge)
#  - Gauche : compteur (vitesse simulée) + total alertes
#  - Droite : compte-tours/activité IA + voyant rouge "Obstacle"
#  - Bas    : Start / Pause / Stop / Replay / Statistiques / Nuit-Jour
#  - Gauche (dock) : Historique défilant (logs & alertes)
#  - Bas droite   : Mini carte POI (icônes 🏫 🏥 ⚽ qui s’allument)
#  - Thread QThread pour la détection, signaux vers le GUI (pas de blocage)
# =============================================================================

import sys                              # Accès argv/exit
import time                             # Petits délais
from PyQt5.QtCore import (Qt, QThread, pyqtSignal, pyqtSlot, QTimer)  # Signaux/Threads/Timers
from PyQt5.QtGui import (QPixmap, QImage, QIcon, QFont)               # Images/icônes/polices
from PyQt5.QtWidgets import (                                          
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QListWidget,
    QFileDialog, QProgressBar, QDockWidget, QAction, QTabWidget,
    QStyleFactory, QShortcut
)
import cv2                              # Conversion BGR → RGB (affichage)
import numpy as np                      # Typage images
import matplotlib
matplotlib.use("Agg")                   # Backend non interactif (redessin manuel)
from matplotlib.figure import Figure    # Figure matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# Import du moteur
import core                              # Notre module logique

# ------------------------------- STYLES QSS -----------------------------------

# Thème clair (Jour)
LIGHT_QSS = """
QMainWindow { background: #f3f4f6; }
QLabel { color: #111827; }
QFrame#VideoFrame { border: 6px solid #2D9CDB; border-radius: 20px; background: #000; }
QFrame#GaugeFrame { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 16px; }
QLabel#BigNumber { font-size: 28px; font-weight: 700; color: #111827; }
QLabel#GaugeTitle { font-size: 14px; color: #4b5563; }
QPushButton { background: #2563eb; color: white; padding: 10px 16px; border-radius: 10px; }
QPushButton:disabled { background: #9ca3af; color: #f9fafb; }
QListWidget { background: #ffffff; border: 1px solid #e5e7eb; border-radius: 10px; }
QProgressBar { border: 1px solid #e5e7eb; border-radius: 8px; text-align: center; }
QProgressBar::chunk { background-color: #10b981; }
QFrame#AlertLed { background: #333; border-radius: 10px; }
"""

# Thème sombre (Nuit)
DARK_QSS = """
QMainWindow { background: #0b0f1a; }
QLabel { color: #e5e7eb; }
QFrame#VideoFrame { border: 6px solid #2D9CDB; border-radius: 20px; background: #000; }
QFrame#GaugeFrame { background: #0f172a; border: 1px solid #1f2937; border-radius: 16px; }
QLabel#BigNumber { font-size: 28px; font-weight: 700; color: #e5e7eb; }
QLabel#GaugeTitle { font-size: 14px; color: #9ca3af; }
QPushButton { background: #1d4ed8; color: white; padding: 10px 16px; border-radius: 10px; }
QPushButton:disabled { background: #334155; color: #94a3b8; }
QListWidget { background: #0f172a; border: 1px solid #1f2937; border-radius: 10px; color: #e5e7eb; }
QProgressBar { border: 1px solid #1f2937; border-radius: 8px; text-align: center; color: #e5e7eb; }
QProgressBar::chunk { background-color: #22c55e; }
QFrame#AlertLed { background: #333; border-radius: 10px; }
"""

# ---------------------- UTILITAIRE : OpenCV → QPixmap -------------------------

def cv2_to_qpixmap(frame_bgr):
    """
    Convertit une image BGR (OpenCV) en QPixmap (affichable dans un QLabel).
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)                 # BGR→RGB
    h, w, ch = frame_rgb.shape                                             # Dimensions
    bytes_per_line = ch * w                                                # Stride
    qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)  # QImage
    return QPixmap.fromImage(qimg)                                         # → QPixmap
# ----------------------------- WORKER DÉTECTION -------------------------------

class DetectionWorker(QThread):
    """
    Thread qui lit une source vidéo (webcam ou fichier), exécute YOLO,
    gère tailgating/alertes/POI, et émet des signaux vers l'UI.
    """
    newFrame = pyqtSignal(object)           # Image BGR annotée (np.ndarray)
    alertEvent = pyqtSignal(dict)           # {type, distance_m, ts, extra}
    distanceUpdate = pyqtSignal(float)      # Distance voiture la plus proche (ou None)
    poiUpdate = pyqtSignal(dict)            # {"school":bool,"hospital":bool,"stadium":bool}
    speedUpdate = pyqtSignal(float)         # Vitesse simulée km/h
    activityUpdate = pyqtSignal(int)        # "Compte-tours" IA (0-100)

    def __init__(self, source=0, parent=None):
        super().__init__(parent)
        self.source = source                # 0 (webcam) ou chemin vidéo
        self._running = True                # Flag boucle active
        self._paused = False                # Flag pause
        self.state = core.DetectionState()  # État de détection
        self._last_poi_check = 0.0          # Throttle POI
        self._poi_interval = 5.0            # Intervalle POI (s)
        self._pos_gen = core.simulated_position_generator() # GPS simulé
        self._prev_pos = None               # Dernière position (lat,lon)
        self._prev_pos_ts = None            # TS de la dernière position
        self._yolo = None                   # Cache modèle (chargé dans run)

    def pause(self, v=True):
        """Met en pause (v=True) ou reprend (v=False) le traitement vidéo."""
        self._paused = v

    def stop(self):
        """Demande l'arrêt propre du thread."""
        self._running = False

    def _update_speed(self):
        """
        Calcule une vitesse simulée en km/h à partir du générateur GPS.
        Émet toujours un signal speedUpdate, même si estimation minimale.
        """
        pos_str = next(self._pos_gen)                     # "lat,lon"
        lat, lon = map(float, pos_str.split(","))
        now = time.time()
        speed_kmh = 0.0
        if self._prev_pos is not None and self._prev_pos_ts is not None:
            d = core.haversine_distance_m(self._prev_pos[0], self._prev_pos[1], lat, lon)  # m
            dt = max(1e-3, now - self._prev_pos_ts)     # s (évite /0)
            speed_kmh = (d / dt) * 3.6                  # m/s → km/h
        self._prev_pos = (lat, lon)                     # Mémorise position
        self._prev_pos_ts = now                         # Mémorise temps
        self.speedUpdate.emit(float(speed_kmh))         # Envoie vitesse
        return pos_str                                  # Retourne position pour POI

    def run(self):
        """Boucle principale : capture → YOLO → événements → signaux GUI"""
        cap = cv2.VideoCapture(self.source)             # Ouvre la source
        if not cap.isOpened():                          # Si échec d'ouverture
            core.logger.error("Impossible d'ouvrir la source vidéo")  # Log
            return

        self._yolo = core.load_model()                  # Charge YOLO (singleton)

        # Message d'accueil
        core.logger.info("Détection démarrée (q=Stop via UI).")

        while self._running:
            if self._paused:
                time.sleep(0.05)
                continue

            ret, frame = cap.read()                     # Lit une frame
            if not ret:                                 # Fin de fichier ou erreur
                break

            frame_h, frame_w = frame.shape[:2]          # Taille image
            results = self._yolo(frame, imgsz=640, verbose=False)  # Inference YOLO
            annotated = results[0].plot()               # Dessins YOLO (bbox/labels)

            labels_this_frame = set()                   # Labels vus pour buffers
            nearest_car_dist = None                     # Distance min voiture

            # Activité IA = nb boxes → 0..100 (pour "compte-tours")
            boxes_count = 0

            for r in results:
                boxes_count += len(getattr(r, "boxes", []))  # Compte boîtes
                for box in r.boxes:
                    cls = int(box.cls[0])                   # ID classe YOLO
                    label = self._yolo.names[cls]           # Nom lisible
                    labels_this_frame.add(label)            # Marque vu

                    # Incrémente compteur anti-faux positifs
                    self.state.detection_buffer[label] = self.state.detection_buffer.get(label, 0) + 1

                    # Cas "car" → estimations distance + "tailgating"
                    if label == "car":
                        dist_m = core.estimate_distance_from_bbox(box, frame_h)  # Estimation distance
                        # Mémorise la plus proche pour la jauge distance
                        if nearest_car_dist is None or dist_m < nearest_car_dist:
                            nearest_car_dist = dist_m

                        # Centre bbox en X (détermine si "dans notre voie" approx.)
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cx = (x1 + x2) / 2
                        lane_center_threshold = frame_w * 0.25  # Largeur tolérée

                        # Si proche et dans l'axe → incrémente buffer tailgating
                        if dist_m <= core.TAILGATE_THRESHOLD_METERS and abs(cx - frame_w / 2) <= lane_center_threshold:
                            self.state.tailgate_buffer += 1
                        else:
                            self.state.tailgate_buffer = max(0, self.state.tailgate_buffer - 1)

                        # Si tailgating persistant → message calmant + capture + log
                        if self.state.tailgate_buffer >= core.TAILGATE_FRAMES_REQUIRED:
                            core.calm_driver_notify(self.state)  # Message + son + log
                            cv2.imwrite(f"{core.CAPTURE_DIR}/tailgate_{int(time.time())}.jpg", frame)  # Capture
                            core.log_alert(self.state, "tailgating", f"(dist={int(dist_m)}m)")         # Log alerte
                            self.alertEvent.emit({"type": "tailgating", "distance_m": float(dist_m), "ts": time.time(), "extra": ""})
                            self.state.tailgate_buffer = 0       # Reset anti-spam

                    # Cas général : alerte "person" ou "car" si frames consécutives suffisantes
                    if label in ["person", "car"] and self.state.detection_buffer.get(label, 0) >= core.ALERT_THRESHOLD_FRAMES:
                        core.log_alert(self.state, label)         # Compte + log + historique
                        core.play_sound(core.ALERT_SOUND)         # Son d'alerte
                        cv2.imwrite(f"{core.CAPTURE_DIR}/alert_{self.state.alert_count}_{label}.jpg", frame)  # Capture
                        self.alertEvent.emit({"type": label, "distance_m": float(nearest_car_dist or 0.0), "ts": time.time(), "extra": ""})
                        self.state.detection_buffer[label] = 0    # Reset buffer label

            # Décrément des buffers pour labels absents de cette frame
            for lbl in list(self.state.detection_buffer.keys()):
                if lbl not in labels_this_frame:
                    self.state.detection_buffer[lbl] = max(0, self.state.detection_buffer[lbl] - 1)

            # Émet l'image annotée pour l'UI
            self.newFrame.emit(annotated)

            # Émet la distance de la voiture la plus proche (ou None) pour la jauge
            if nearest_car_dist is not None:
                self.distanceUpdate.emit(float(nearest_car_dist))
            else:
                self.distanceUpdate.emit(float('nan'))  # UI décidera comment l'afficher

            # Émet un niveau d'activité IA (0..100) en fonction du nombre de boxes
            activity = max(0, min(100, int(boxes_count * 5)))    # Simple scaling
            self.activityUpdate.emit(activity)

            # Met à jour vitesse simulée + récupère position pour POI
            user_pos_str = self._update_speed()

            # Vérification POI périodique
            now_ts = time.time()
            if now_ts - self._last_poi_check > self._poi_interval:
                self._last_poi_check = now_ts
                poi_leds = core.check_and_notify_poi(user_pos_str, core.GOOGLE_API_KEY, self.state)
                self.poiUpdate.emit(poi_leds)

        cap.release()                                  # Libère la source vidéo
        core.logger.info("Détection arrêtée.")

# ----------------------------- CANVAS STATISTIQUES ----------------------------

class StatsCanvas(FigureCanvas):
    """
    Petit canvas matplotlib redessiné par timer, basé sur state.alert_history.
    """
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(4, 2.6), dpi=100)   # Crée la figure
        super().__init__(self.fig)                     # Init base
        self.ax = self.fig.add_subplot(111)            # Axe unique

    def redraw(self, alert_history):
        self.ax.clear()                               # Nettoie l'axe
        if alert_history:
            t0 = alert_history[0][0]                  # Référence temps
            xs = [t - t0 for (t, _) in alert_history] # Temps relatifs (s)
            ys = [c for (_, c) in alert_history]      # Compteur d'alertes
            self.ax.plot(xs, ys, marker="o")          # Courbe simple
            self.ax.set_title("Évolution des alertes")
            self.ax.set_xlabel("Temps (s)")
            self.ax.set_ylabel("Alertes")
            self.ax.grid(True, linestyle="--", alpha=0.3)
        else:
            self.ax.text(0.5, 0.5, "Aucune alerte pour l’instant",
                          ha="center", va="center", transform=self.ax.transAxes)
        self.draw_idle()                              # Redessin asynchrone
# ------------------------------ FENÊTRE PRINCIPALE ----------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sécurité Routière — Cockpit")       # Titre
        self.setMinimumSize(1200, 720)                           # Taille mini
        self.setStyleSheet(DARK_QSS)                             # Thème par défaut (nuit)
        self._night_mode = True                                  # État thème

        # --------- WIDGETS PRINCIPAUX ---------

        # Gauche : cadre "compteurs"
        self.leftFrame = QFrame()
        self.leftFrame.setObjectName("GaugeFrame")
        leftLayout = QVBoxLayout(self.leftFrame)
        # Titre vitesse
        self.lblSpeedTitle = QLabel("Vitesse (km/h)")
        self.lblSpeedTitle.setObjectName("GaugeTitle")
        self.lblSpeed = QLabel("0")               # Grand chiffre de vitesse
        self.lblSpeed.setObjectName("BigNumber")
        self.lblAlertsTitle = QLabel("ALERTES")
        self.lblAlertsTitle.setObjectName("GaugeTitle")
        self.lblAlerts = QLabel("0")
        self.lblAlerts.setObjectName("BigNumber")
        leftLayout.addWidget(self.lblSpeedTitle)
        leftLayout.addWidget(self.lblSpeed)
        leftLayout.addSpacing(10)
        leftLayout.addWidget(self.lblAlertsTitle)
        leftLayout.addWidget(self.lblAlerts)
        leftLayout.addStretch(1)

        # Centre : vidéo avec anneau LED (frame + label image)
        self.videoFrame = QFrame()
        self.videoFrame.setObjectName("VideoFrame")
        centerLayout = QVBoxLayout(self.videoFrame)
        self.videoLabel = QLabel("Vidéo…")
        self.videoLabel.setAlignment(Qt.AlignCenter)
        centerLayout.addWidget(self.videoLabel)
        # Jauge distance
        self.lblDistTitle = QLabel("Distance estimée (m)")
        self.lblDistTitle.setObjectName("GaugeTitle")
        self.distanceBar = QProgressBar()
        self.distanceBar.setRange(0, 100)          # 0=proche (danger), 100=loin (sûr)
        self.distanceBar.setValue(100)
        centerLayout.addWidget(self.lblDistTitle)
        centerLayout.addWidget(self.distanceBar)

        # Droite : activité IA + voyant obstacle + mini carte POI
        self.rightFrame = QFrame()
        self.rightFrame.setObjectName("GaugeFrame")
        rightLayout = QVBoxLayout(self.rightFrame)
        # Activité IA
        self.lblActTitle = QLabel("Activité IA (%)")
        self.lblActTitle.setObjectName("GaugeTitle")
        self.lblActivity = QLabel("0")
        self.lblActivity.setObjectName("BigNumber")
        # Voyant obstacle
        self.alertLed = QFrame()
        self.alertLed.setObjectName("AlertLed")
        self.alertLed.setFixedSize(20, 20)
        self._set_alert_led(False)                  # LED éteinte au démarrage
        # POI
        self.lblPoiTitle = QLabel("POI à proximité")
        self.lblPoiTitle.setObjectName("GaugeTitle")
        self.poiSchool = QLabel("🏫")
        self.poiHospital = QLabel("🏥")
        self.poiStadium = QLabel("⚽")
        poiRow = QHBoxLayout()
        poiRow.addWidget(self.poiSchool)
        poiRow.addWidget(self.poiHospital)
        poiRow.addWidget(self.poiStadium)

        rightLayout.addWidget(self.lblActTitle)
        rightLayout.addWidget(self.lblActivity)
        rightLayout.addSpacing(10)
        rightLayout.addWidget(QLabel("Obstacle"))
        rightLayout.addWidget(self.alertLed)
        rightLayout.addSpacing(10)
        rightLayout.addWidget(self.lblPoiTitle)
        rightLayout.addLayout(poiRow)
        rightLayout.addStretch(1)

        # Bandeau bas : boutons
        self.btnStart = QPushButton("Start")
        self.btnPause = QPushButton("Pause")
        self.btnStop = QPushButton("Stop")
        self.btnReplay = QPushButton("Replay")
        self.btnStats = QPushButton("Statistiques")
        self.btnTheme = QPushButton("Mode Nuit/Jour")
        self.btnPause.setEnabled(False)
        self.btnStop.setEnabled(False)
        bottomRow = QHBoxLayout()
        for btn in [self.btnStart, self.btnPause, self.btnStop, self.btnReplay, self.btnStats, self.btnTheme]:
            bottomRow.addWidget(btn)

        # Layout central (gauche | centre | droite)
        center = QWidget()
        grid = QGridLayout(center)
        grid.addWidget(self.leftFrame,   0, 0)
        grid.addWidget(self.videoFrame,  0, 1)
        grid.addWidget(self.rightFrame,  0, 2)
        grid.addLayout(bottomRow,        1, 0, 1, 3)  # Ligne boutons sur 3 colonnes
        self.setCentralWidget(center)

        # Dock gauche : historique
        self.historyDock = QDockWidget("Historique")
        self.historyList = QListWidget()
        self.historyDock.setWidget(self.historyList)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.historyDock)

        # Onglet/statistiques (dans un dock pour l’ouvrir quand on veut)
        self.statsDock = QDockWidget("Statistiques")
        self.statsCanvas = StatsCanvas()
        statsWrap = QWidget()
        vstats = QVBoxLayout(statsWrap)
        vstats.addWidget(self.statsCanvas)
        self.statsDock.setWidget(statsWrap)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.statsDock)
        self.statsDock.hide()  # Masqué par défaut

        # --------- ÉTAT & TIMERS ---------
        self.worker = None                          # Pas de worker tant qu’on n’a pas démarré
        self._ring_state = "calm"                   # État anneau LED ("calm"|"warn"|"danger")
        self._alerts_total = 0                      # Compteur total UI (répliqué depuis state)
        self._stats_timer = QTimer(self)            # Timer pour redessiner Stats
        self._stats_timer.timeout.connect(self._refresh_stats)
        self._stats_timer.start(1000)               # Toutes les 1s

        # --------- SIGNAUX BOUTONS ---------
        self.btnStart.clicked.connect(self.on_start)
        self.btnPause.clicked.connect(self.on_pause_resume)
        self.btnStop.clicked.connect(self.on_stop)
        self.btnReplay.clicked.connect(self.on_replay)
        self.btnStats.clicked.connect(self.on_toggle_stats)
        self.btnTheme.clicked.connect(self.on_toggle_theme)

        # --------- RACCOURCIS CLAVIER ---------
        QShortcut(Qt.Key_S, self, activated=self.on_start)          # S : Start
        QShortcut(Qt.Key_P, self, activated=self.on_pause_resume)   # P : Pause
        QShortcut(Qt.Key_Q, self, activated=self.on_stop)           # Q : Stop
        QShortcut(Qt.Key_R, self, activated=self.on_replay)         # R : Replay
        QShortcut(Qt.Key_T, self, activated=self.on_toggle_theme)   # T : Nuit/Jour

    # --------------------- OUTILS UI (LED, anneau, jauge) ---------------------

    def _set_alert_led(self, on: bool):
        """Allume/éteint le voyant rouge 'Obstacle'."""
        if on:
            self.alertLed.setStyleSheet("QFrame#AlertLed { background:#EB5757; border-radius:10px; }")
        else:
            self.alertLed.setStyleSheet("QFrame#AlertLed { background:#333; border-radius:10px; }")

    def _set_video_ring(self, state: str):
        """
        Met à jour la couleur de l'anneau LED autour de la vidéo :
        - 'calm'  → bleu
        - 'warn'  → jaune
        - 'danger'→ rouge
        """
        color = {"calm": "#2D9CDB", "warn": "#F2C94C", "danger": "#EB5757"}.get(state, "#2D9CDB")
        self.videoFrame.setStyleSheet(f"QFrame#VideoFrame {{ border: 6px solid {color}; border-radius:20px; background:#000; }}")
        self._ring_state = state

    def _set_distance_bar(self, dist_m: float):
        """
        Met à jour la jauge de distance (0..100).
        100 = loin/sûr ; 0 = danger.
        """
        if dist_m != dist_m:  # test NaN
            self.distanceBar.setValue(100)
            self._set_video_ring("calm")
            return
        level = max(0, min(100, int((dist_m / 25.0) * 100)))  # 25m → 100%
        self.distanceBar.setValue(level)
        # Seuils de couleur/anneau
        if dist_m < 8:
            self._set_video_ring("danger")
        elif dist_m < 15:
            self._set_video_ring("warn")
        else:
            self._set_video_ring("calm")

    # -------------------------- GESTION STATS UI -------------------------------

    def _refresh_stats(self):
        """Redessine la courbe Stats à partir de l'historique du state (si worker actif)."""
        if self.worker is not None:
            self.statsCanvas.redraw(self.worker.state.alert_history)

    # ------------------------------ SLOTS WORKER -------------------------------

    @pyqtSlot(object)
    def on_new_frame(self, frame_bgr):
        """Réception d'une frame annotée → affichage au centre."""
        self.videoLabel.setPixmap(cv2_to_qpixmap(frame_bgr))

    @pyqtSlot(dict)
    def on_alert_event(self, info):
        """Ajoute une entrée d'historique + met à jour total + LED obstacle."""
        ts_str = core.human_time(info.get("ts"))
        typ = info.get("type")
        dist = info.get("distance_m")
        line = f"[{ts_str}] Alerte : {typ}"
        if dist and dist == dist:  # pas NaN
            line += f" (dist≈{int(dist)}m)"
        self.historyList.addItem(line)
        # Met à jour le total à partir du state (source de vérité)
        if self.worker is not None:
            self._alerts_total = self.worker.state.alert_count
        self.lblAlerts.setText(str(self._alerts_total))
        # LED rouge ON pour toute alerte de type obstacle
        self._set_alert_led(True)
        # Éteindre après un court délai (petit effet)
        QTimer.singleShot(800, lambda: self._set_alert_led(False))

    @pyqtSlot(float)
    def on_distance_update(self, dist_m):
        """Jauge distance + anneau LED."""
        self._set_distance_bar(dist_m)

    @pyqtSlot(dict)
    def on_poi_update(self, leds):
        """Allume/éteint les icônes POI."""
        self.poiSchool.setText("🏫" + (" ✅" if leds.get("school") else ""))
        self.poiHospital.setText("🏥" + (" ✅" if leds.get("hospital") else ""))
        self.poiStadium.setText("⚽" + (" ✅" if leds.get("stadium") else ""))

    @pyqtSlot(float)
    def on_speed_update(self, speed_kmh):
        """Affiche la vitesse simulée (arrondie)."""
        self.lblSpeed.setText(str(int(speed_kmh)))

    @pyqtSlot(int)
    def on_activity_update(self, val):
        """Met à jour l'activité IA (0..100)."""
        self.lblActivity.setText(str(val))
    # ---------------------------- COMMANDES BOUTONS ----------------------------

    def _connect_worker_signals(self):
        """Connecte tous les signaux du worker aux slots de la fenêtre."""
        self.worker.newFrame.connect(self.on_new_frame)
        self.worker.alertEvent.connect(self.on_alert_event)
        self.worker.distanceUpdate.connect(self.on_distance_update)
        self.worker.poiUpdate.connect(self.on_poi_update)
        self.worker.speedUpdate.connect(self.on_speed_update)
        self.worker.activityUpdate.connect(self.on_activity_update)

    def _set_ui_running(self, running: bool):
        """Active/désactive les boutons selon l'état."""
        self.btnStart.setEnabled(not running)
        self.btnPause.setEnabled(running)
        self.btnStop.setEnabled(running)
        self.btnReplay.setEnabled(not running)

    def on_start(self):
        """Démarre la détection sur la webcam intégrée (source=0)."""
        if self.worker is not None:
            return
        self.worker = DetectionWorker(source=0)
        self._connect_worker_signals()
        self._alerts_total = 0
        self.lblAlerts.setText("0")
        self._set_ui_running(True)
        self.worker.start()  # Lance le thread

    def on_pause_resume(self):
        """Bascule pause/reprise."""
        if not self.worker:
            return
        if self.worker._paused:
            self.worker.pause(False)
            self.btnPause.setText("Pause")
        else:
            self.worker.pause(True)
            self.btnPause.setText("Reprendre")

    def on_stop(self):
        """Arrête proprement la détection."""
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.worker = None
        self._set_ui_running(False)
        self.btnPause.setText("Pause")
        # Reset voyants
        self._set_alert_led(False)
        self._set_video_ring("calm")
        self.distanceBar.setValue(100)

    def on_replay(self):
        """Choisit un fichier vidéo et démarre la détection dessus (mode Replay)."""
        if self.worker is not None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Choisir une vidéo", "", "Vidéos (*.mp4 *.avi *.mkv)")
        if not path:
            return
        self.worker = DetectionWorker(source=path)
        self._connect_worker_signals()
        self._alerts_total = 0
        self.lblAlerts.setText("0")
        self._set_ui_running(True)
        self.worker.start()

    def on_toggle_stats(self):
        """Affiche/masque le dock des statistiques."""
        if self.statsDock.isHidden():
            self.statsDock.show()
        else:
            self.statsDock.hide()

    def on_toggle_theme(self):
        """Bascule entre thème Nuit et Jour."""
        self._night_mode = not self._night_mode
        self.setStyleSheet(DARK_QSS if self._night_mode else LIGHT_QSS)

# ---------------------------------- MAIN --------------------------------------

def main():
    app = QApplication(sys.argv)         # Crée l'application Qt
    app.setStyle(QStyleFactory.create("Fusion"))  # Style cohérent cross-platform
    win = MainWindow()                   # Fenêtre principale
    win.show()                           # Affiche
    sys.exit(app.exec_())                # Boucle Qt

if __name__ == "__main__":
    main()
