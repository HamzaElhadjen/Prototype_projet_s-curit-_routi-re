# core.py
# =============================================================================
#  MOTEUR "Sécurité Routière" — logique réutilisable par une interface GUI
#  - Ce module contient :
#      * les constantes et variables d'état
#      * le logger avec rotation
#      * les utilitaires audio et de calcul (distance, haversine, etc.)
#      * la simulation GPS + POI (Google Places simulé par défaut)
#      * un chargeur de modèle YOLO et une classe d'état
#  - Il NE contient pas d'affichage OpenCV (pas de cv2.imshow ni waitKey)
#  - Il est pensé pour être appelé depuis main.py (PyQt5)
# =============================================================================

# ------------------------------ IMPORTS ---------------------------------------
import os                              # Gestion fichiers/dossiers
import time                            # Timestamps / temporisations
import math                            # Fonctions math (haversine, etc.)
import datetime                        # Horodatages lisibles
import threading                       # Jouer les sons sans bloquer
import logging                         # Journalisation standard
from logging.handlers import RotatingFileHandler  # Rotation de logs
import requests                        # (Optionnel) API Google Places si activée
import cv2                             # OpenCV (traitement image, encodage)
from playsound import playsound        # Lecture audio (bloquant → thread)
from ultralytics import YOLO           # Modèle YOLOv8 (détection)

# --------------------------- CONFIGURATION GLOBALE ----------------------------

# Dossier des captures d'écran (créé s'il n'existe pas)
CAPTURE_DIR = "captures"                               # Dossier où enregistrer les images
os.makedirs(CAPTURE_DIR, exist_ok=True)                # Création sûre (si absent)

# Fichier de logs + rotation
LOG_FILE = "log.txt"                                   # Chemin du fichier de log

# Sons principaux (si non trouvés, l'appel sera ignoré sans planter)
ALERT_SOUND = "alert.mp3"                              # Son pour alerte confirmée
CALM_SOUND  = "calm_alert.mp3"                         # Son "calmant" par défaut
POI_SOUND   = "poi_alert.mp3"                          # Son générique POI

# Variantes de sons calmants (une est choisie pseudo-aléatoirement si dispo)
CALM_SOUNDS = [                                        # Liste des sons calmants
    "calm1.mp3",
    "calm2.mp3",
    "calm3.mp3",
    "cal4.mp3",
    "calm5.mp3",
]

# Sons spécifiques POI (si fichiers manquants → fallback POI_SOUND)
POI_AUDIO_MAP = {                                      # Mapping type → sons
    "school":   ["school1.mp3", "school2.mp3"],
    "hospital": ["hospital1.mp3", "hospital2.mp3"],
    "stadium":  ["stadium1.mp3",  "stadium2.mp3"],
}

# Paramètres détection/filtres
ALERT_THRESHOLD_FRAMES    = 5                           # Frames consécutives avant alerte "person"/"car"
TAILGATE_THRESHOLD_METERS = 8.0                         # Distance max pour "tailgating" (voiture trop proche)
TAILGATE_FRAMES_REQUIRED  = 3                           # Frames consécutives avant message calmant

# Estimation heuristique distance (à partir de la hauteur de bbox/hauteur image)
DISTANCE_SCALE = 2.5                                    # Constante proportionnelle
MIN_DISTANCE_M = 1.0                                    # Distance mini retournée
MAX_DISTANCE_M = 200.0                                  # Distance maxi retournée

# POI / Google Places (simulation si None)
GOOGLE_API_KEY = None                                   # None = simulation (pas d'appel réseau)
POI_SEARCH_RADIUS_METERS = 500                          # Rayon de recherche autour de la position
POI_TYPES = ["school", "hospital", "stadium"]           # Types monitorés
POI_NOTIFY_DISTANCE_METERS = 200                        # Seuil d'annonce (m)

# ------------------------------ LOGGER ----------------------------------------

# Création d'un logger dédié (utilisé par GUI via un handler Qt optionnel)
logger = logging.getLogger("RoadSafety")                # Nom du logger
logger.setLevel(logging.INFO)                           # Niveau d'info

# Gestionnaire avec rotation (évite un log qui grossit trop)
_rot_handler = RotatingFileHandler(                     # Handler fichier rotatif
    LOG_FILE,
    maxBytes=200_000,                                   # ~200 Ko avant rotation
    backupCount=5,                                      # Conserve 5 archives
    encoding="utf-8",
)
_rot_handler.setFormatter(logging.Formatter(            # Format lisible
    "[%(asctime)s] %(levelname)s: %(message)s"
))
if not any(isinstance(h, RotatingFileHandler) for h in logger.handlers):
    logger.addHandler(_rot_handler)                     # Évite double ajout

# ------------------------------ AUDIO -----------------------------------------

def choose_from_files(files, default=None):
    """
    Retourne un fichier existant depuis une liste (rotation simple par timestamp).
    - files   : list[str] chemins candidats
    - default : str|None fallback si aucun fichier existant
    """
    existing = [f for f in files if os.path.exists(f)]  # Garde les fichiers présents
    if not existing:                                    # Si aucun disponible
        return default                                  # Renvoie le défaut (ou None)
    idx = int(time.time()) % len(existing)              # Index pseudo-aléatoire
    return existing[idx]                                # Fichier choisi

def play_sound(path):
    """
    Joue un son NON-bloquant (dans un thread daemon).
    - path : str chemin du fichier audio (ignorer si absent)
    """
    if not path:                                        # Aucun chemin
        return                                          # Ne rien faire
    if not os.path.exists(path):                        # Fichier inexistant
        logger.warning(f"Son introuvable : {path}")     # Log warning
        return                                          # Sortie silencieuse
    threading.Thread(target=lambda: playsound(path),    # Lance playsound
                     daemon=True).start()               # Thread daemon = auto-stop

def select_poi_audio(types_list):
    """
    Sélectionne un son adapté au type de POI détecté (école/hôpital/stade).
    - types_list : list[str] types fournis par Google Places (ou simulation)
    """
    for t in POI_TYPES:                                 # Itère types connus
        if t in types_list and t in POI_AUDIO_MAP:      # Si type reconnu
            return choose_from_files(POI_AUDIO_MAP[t],  # Choix fichier
                                      default=POI_SOUND) or POI_SOUND
    return POI_SOUND                                    # Fallback générique

# ------------------------------ OUTILS ----------------------------------------

def log_alert(state, obj_name, extra=""):
    """
    Incrémente le compteur d'alertes, journalise et pousse l'historique.
    - state    : DetectionState instance (compteurs/historique)
    - obj_name : str  (ex: "person", "car", "tailgating", "poi:school")
    - extra    : str  (ex: "(dist=7m)")
    """
    state.alert_count += 1                              # +1 alerte globale
    msg = f"Alerte {state.alert_count} : {obj_name} {extra}".strip()  # Message
    logger.info(msg)                                    # Log dans fichier
    print(msg)                                          # Console (utile démo)
    state.alert_history.append((time.time(),            # Historique
                                state.alert_count))

def estimate_distance_from_bbox(box, frame_height):
    """
    Estime une distance en mètres à partir de la hauteur de la bbox YOLO.
    - box          : ultralytics.yolo.engine.results.Boxes item
    - frame_height : int hauteur image (px)
    """
    try:                                                # Essai conversion numpy
        xy = box.xyxy[0].cpu().numpy()
    except Exception:                                   # Fallback si pas numpy
        xy = box.xyxy[0]
    x1, y1, x2, y2 = map(int, xy)                       # Coordonnées entières
    bbox_h = max(1, y2 - y1)                            # Hauteur bbox (>=1)
    ratio = bbox_h / max(1, frame_height)               # Ratio relatif
    dist = DISTANCE_SCALE / max(ratio, 1e-6)            # Inverse du ratio
    return max(MIN_DISTANCE_M, min(MAX_DISTANCE_M,      # Bornage
                                   dist))

def haversine_distance_m(lat1, lon1, lat2, lon2):
    """
    Distance orthodromique (m) entre deux points GPS (formule de Haversine).
    """
    R = 6371e3                                          # Rayon Terre (m)
    phi1, phi2 = math.radians(lat1), math.radians(lat2) # Latitudes rad
    dphi    = math.radians(lat2 - lat1)                 # Δlat
    dlambda = math.radians(lon2 - lon1)                 # Δlon
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))    # Angle central
    return R * c  # ------------------------------- POI ------------------------------------------

def get_nearby_places(api_key, location, radius, types, poi_cache):
    """
    Renvoie une liste de POI autour d'une position.
    - api_key   : str|None → None = mode simulation (pas d'appel réseau)
    - location  : "lat,lon"
    - radius    : int rayon en mètres
    - types     : list[str] types de lieux
    - poi_cache : dict cache simple pour limiter les appels (clé=(loc,types))
    """
    if api_key is None:
        # Simulation simple : génère 3 POI autour de la position actuelle
        lat, lon = map(float, location.split(","))
        return [
            {"name": "École Simone",  "geometry": {"location": {"lat": lat + 0.001,  "lng": lon + 0.001}},  "types": ["school"]},
            {"name": "Hôpital Saint", "geometry": {"location": {"lat": lat + 0.002,  "lng": lon - 0.0005}}, "types": ["hospital"]},
            {"name": "Stade Central", "geometry": {"location": {"lat": lat - 0.0015, "lng": lon + 0.002}},  "types": ["stadium"]},
        ]

    # Avec API réelle (optionnel) — simple cache 60s
    key = (location, tuple(types))
    now = time.time()
    if key in poi_cache and now - poi_cache[key]["ts"] < 60:
        return poi_cache[key]["data"]

    url = (
        "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        f"?location={location}&radius={radius}&types={'|'.join(types)}&key={api_key}"
    )
    try:
        r = requests.get(url, timeout=5)
        data = r.json().get("results", [])
        poi_cache[key] = {"data": data, "ts": now}
        return data
    except Exception as e:
        logger.error(f"Erreur API Google Places : {e}")
        return []

def check_and_notify_poi(user_location_str, api_key, state):
    """
    Vérifie les POI proches et déclenche (à <200 m) une alerte + son.
    - user_location_str : "lat,lon"
    - api_key           : str|None
    - state             : DetectionState (LAST_POI_NOTIFIED, cache, etc.)
    Retourne un dict d'états pour le GUI : {"school":bool,"hospital":bool,"stadium":bool}
    """
    u_lat, u_lon = map(float, user_location_str.split(","))
    places = get_nearby_places(api_key, user_location_str,
                               POI_SEARCH_RADIUS_METERS, POI_TYPES, state.poi_cache)
    # Initialisation des voyants pour le mini-paneau POI
    led_states = {"school": False, "hospital": False, "stadium": False}

    for p in places:
        name  = p.get("name", "Lieu")                                   # Nom lisible
        p_lat = p.get("geometry", {}).get("location", {}).get("lat")    # Lat POI
        p_lon = p.get("geometry", {}).get("location", {}).get("lng")    # Lon POI
        if p_lat is None or p_lon is None:
            continue                                                    # Coord manquantes → skip

        dist_m = haversine_distance_m(u_lat, u_lon, p_lat, p_lon)       # Distance POI
        types  = p.get("types", [])                                     # Types POI
        # Active les voyants si POI "proche" (même si pas de notification fraîche)
        for t in POI_TYPES:
            if t in types and dist_m <= POI_NOTIFY_DISTANCE_METERS:
                led_states[t] = True

        # Anti-spam : ne notifie pas le même POI trop souvent
        poi_id = name
        now_ts = time.time()
        if dist_m <= POI_NOTIFY_DISTANCE_METERS and (now_ts - state.last_poi_notified.get(poi_id, 0)) > 30:
            msg = f"⚠️ Approche : {name} à ~{int(dist_m)} m"
            logger.info(msg)
            print(msg)
            # Son spécifique (ou générique)
            sound_path = select_poi_audio(types)
            play_sound(sound_path)
            # Log + historique
            log_alert(state, f"poi:{types[0] if types else 'poi'}", f"({name} ~{int(dist_m)}m)")
            # Mémorise dernière notif
            state.last_poi_notified[poi_id] = now_ts

    return led_states

# ---------------------------- SIMULATION GPS ----------------------------------

def simulated_position_generator():
    """
    Générateur de positions GPS (lat,lon) simulées autour d'Alger.
    Donne une position différente à chaque appel, pour tester POI et vitesse.
    """
    base_lat, base_lon = 36.752778, 3.042222           # Point de base (Alger centre)
    step = 0                                            # Pas de temps (entier)
    while True:                                         # Boucle sans fin
        lat = base_lat + 0.0001 * math.sin(step / 10.0) # Variation sin lat
        lon = base_lon + 0.00015 * math.cos(step / 10.0)# Variation cos lon
        step += 1                                       # Incrémente le pas
        yield f"{lat},{lon}"                            # Renvoie "lat,lon"

# ---------------------------- MODÈLE YOLO / ÉTAT ------------------------------

# Singleton de modèle (pour éviter rechargements coûteux)
_YOLO_SINGLETON = {"model": None}                      # Cache global modèle

def load_model():
    """
    Charge YOLOv8n une seule fois (pattern singleton).
    Retourne l'instance de modèle prête à l'emploi.
    """
    if _YOLO_SINGLETON["model"] is None:                # Si pas encore chargé
        model = YOLO("yolov8n.pt")                      # Charge (auto-download)
        # try:
        #     model.to("cuda")                          # ← Active le GPU si dispo
        # except Exception:
        #     pass
        _YOLO_SINGLETON["model"] = model                # Mémo dans le cache
    return _YOLO_SINGLETON["model"]                     # Retourne le modèle

class DetectionState:
    """
    Structure d'état pour la détection :
    - alert_count        : nombre d'alertes cumulées
    - detection_buffer   : label -> compteur frames consécutives
    - tailgate_buffer    : nombre de frames "voiture trop proche" consécutives
    - last_poi_notified  : dict nomPOI -> timestamp dernière notif
    - poi_cache          : cache de POI (limite les appels)
    - alert_history      : [(timestamp, count)] pour la courbe Stats
    """
    def __init__(self):
        self.alert_count       = 0                      # Compteur global d'alertes
        self.detection_buffer  = {}                     # Anti-faux positifs
        self.tailgate_buffer   = 0                      # Buffer tailgating
        self.last_poi_notified = {}                     # Anti-spam POI
        self.poi_cache         = {}                     # Cache POI (API)
        self.alert_history     = []                     # Historique des alertes

def calm_driver_notify(state):
    """
    Message + son calmants quand tailgating confirmé.
    - state : DetectionState (pour log_alert)
    """
    messages = [
        "Ralentissez… gardez vos distances pour votre sécurité.",
        "La patience évite les accidents. Douceur au volant.",
        "Partager la route, c’est se protéger soi-même.",
        "Un instant de calme peut tout changer. Levez le pied.",
        "Un conducteur serein protège des vies. Restez zen.",
    ]
    msg = messages[int(time.time()) % len(messages)]   # Choix pseudo-aléatoire
    logger.info(f"Message calmant : {msg}")            # Log
    print(f"💬 {msg}")                                  # Console
    sound = choose_from_files(CALM_SOUNDS, default=CALM_SOUND) # Son calmant
    play_sound(sound)                                  # Lecture non bloquante
    log_alert(state, "tailgating:calm")                # Trace comme alerte

# --------------------------- ENCODAGE UTILITAIRE -------------------------------

def encode_frame_to_qimage_bytes(frame_bgr):
    """
    Encodage BGR → JPEG (bytes) utile si on voulait pousser vers du web/stream.
    Non utilisé par PyQt (qui convertit en QImage), mais laissé en utilitaire.
    """
    ok, jpg = cv2.imencode(".jpg", frame_bgr)          # Encode en JPEG
    return jpg.tobytes() if ok else None               # Bytes ou None

# ------------------------------- HELPER TEMPS ---------------------------------

def human_time(ts=None):
    """
    Timestamp → HH:MM:SS pour afficher dans l'historique GUI.
    """
    dt = datetime.datetime.fromtimestamp(ts or time.time())
    return dt.strftime("%H:%M:%S")
                                      # Distance en mètres
