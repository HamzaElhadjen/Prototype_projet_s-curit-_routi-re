"""
================================================================================
MODULE : Prototype Sécurité Routière — Version *ultra commentée* + Statistiques
================================================================================

But du projet :
---------------
- Détection d'objets (personnes, voitures) avec YOLOv8
- Anti-faux positifs grâce à des buffers
- Détection du "tailgating" (voiture trop proche) avec message calmant
- Notifications POI (écoles, hôpitaux, stades) simulées ou via Google Places
- Journalisation professionnelle avec rotation des logs
- Statistiques en direct (nombre d’alertes affiché en graphique Matplotlib)
- Architecture multi-threads pour éviter les blocages (vidéo, son, stats, POI)

Ce code est un **prototype pédagogique** : il n’est pas destiné à la production
immédiate, mais à montrer tes compétences techniques et académiques.
================================================================================
"""

# ------------------------------ IMPORTS ---------------------------------------

# Bibliothèques standards Python
import os            # Gestion du système de fichiers (création dossiers, vérif fichiers)
import threading     # Gestion des threads (exécution parallèle de plusieurs tâches)
import datetime      # Manipulation de dates et heures (utile pour logs, captures)
import time          # Gestion du temps (horodatage, pauses, timestamps)
import math          # Fonctions mathématiques (sin, cos, radians) utiles pour simulation GPS

# Vision par ordinateur, intelligence artificielle et audio
import cv2                              # OpenCV : capture vidéo, affichage, dessin sur image
from ultralytics import YOLO            # Ultralytics : modèle YOLOv8 pour détection en temps réel
from playsound import playsound         # Lecture de fichiers audio (bloquant si direct, donc lancé en thread)
import requests                         # Envoi de requêtes HTTP (API Google Places en mode réel)

# Gestion des logs avec rotation (évite que les fichiers deviennent trop gros)
import logging                          # Système standard de logs Python
from logging.handlers import RotatingFileHandler  # Gestionnaire de rotation automatique des fichiers logs

# Graphiques pour afficher les statistiques en direct
import matplotlib.pyplot as plt         # Matplotlib : génération de graphiques (alertes dans le temps)

# --------------------------- CONFIGURATION GLOBALE ----------------------------

# === Fichiers audio principaux ===
ALERT_SOUND = "alert.mp3"        # Son joué lors d'une alerte confirmée (personne ou voiture détectée)
CALM_SOUND  = "calm_alert.mp3"   # Son par défaut utilisé si aucune variante "calm" n’est trouvée
POI_SOUND   = "poi_alert.mp3"    # Son générique pour signaler un POI si aucun audio spécifique n’existe

# === Variantes de sons "calm" ===
# Liste de plusieurs fichiers sons destinés à apaiser le conducteur lorsqu’il colle une voiture
CALM_SOUNDS = [
    "calm1.mp3",  # « Ralentissez et respirez profondément… »
    "calm2.mp3",  # « La sécurité n’est jamais une perte de temps… »
    "calm3.mp3",  # « Conduire, c’est partager la route… »
    "calm4.mp3",  # « Un instant de patience peut éviter un accident… »
    "calm5.mp3",  # « Un conducteur calme protège sa vie et celle des autres… »
]

# === Sons spécifiques aux POI (Points of Interest) ===
# Mapping : chaque type de POI est associé à plusieurs fichiers audio
POI_AUDIO_MAP = {
    "school":   ["school1.mp3", "school2.mp3"],     # Écoles → prudence enfants
    "hospital": ["hospital1.mp3", "hospital2.mp3"], # Hôpitaux → conduite douce
    "stadium":  ["stadium1.mp3", "stadium2.mp3"],   # Stades → prudence foule/circulation
}

# === Paramètres de détection et anti-faux positifs ===
ALERT_THRESHOLD_FRAMES   = 5     # Nombre minimum de frames consécutives où un objet est détecté avant alerte
TAILGATE_THRESHOLD_METERS = 8.0  # Distance maximale estimée pour considérer que la voiture "colle" (mètres)
TAILGATE_FRAMES_REQUIRED  = 3    # Nombre de frames consécutives de proximité avant notification calmante

# === Paramètres pour les POI (Points d’intérêt) ===
GOOGLE_API_KEY = None            # Clé API Google. None = simulation (pas d’appel réel à Google Places)
POI_SEARCH_RADIUS_METERS = 500   # Rayon de recherche (en mètres) autour de la position GPS
POI_TYPES = ["school", "hospital", "stadium"]  # Types de POI surveillés
POI_NOTIFY_DISTANCE_METERS = 200 # Distance max avant d’envoyer une notification d’approche (en mètres)

# === Estimation heuristique de la distance à partir des bounding boxes ===
DISTANCE_SCALE = 2.5             # Constante proportionnelle pour calcul heuristique
MIN_DISTANCE_M = 1.0             # Distance minimale (plancher en mètres)
MAX_DISTANCE_M = 200.0           # Distance maximale (plafond en mètres)

# === Ressources locales ===
CAPTURE_DIR = "captures"         # Répertoire où sauvegarder les captures d’écran lors d’alertes
LOG_FILE    = "log.txt"          # Nom du fichier log principal

# --------------------------- INITIALISATIONS SYSTÈME --------------------------

# Création du dossier de captures si celui-ci n’existe pas encore
if not os.path.exists(CAPTURE_DIR):   # Vérifie si "captures/" est déjà présent
    os.makedirs(CAPTURE_DIR)          # Le crée automatiquement si absent

# Configuration du logger principal "RoadSafety"
logger = logging.getLogger("RoadSafety")   # Crée un objet logger nommé
logger.setLevel(logging.INFO)              # Niveau de logs : INFO et plus (INFO, WARNING, ERROR)

# Gestionnaire de logs avec rotation automatique
# → Quand log.txt dépasse 200 Ko, il est renommé log.txt.1 et un nouveau est créé
handler = RotatingFileHandler(
    LOG_FILE,            # Chemin du fichier log principal
    maxBytes=200_000,    # Taille max avant rotation (~200 Ko)
    backupCount=5,       # Nombre max de fichiers de sauvegarde conservés (log.txt.1, log.txt.2…)
    encoding="utf-8",    # Encodage pour bien gérer les accents
)

# Format d’une ligne de log (exemple : [2025-09-20 20:10:00] INFO: Alerte voiture)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
handler.setFormatter(formatter)        # Applique ce format au gestionnaire
logger.addHandler(handler)             # Attache le gestionnaire au logger

# --------------------------- VARIABLES D'ÉTAT GLOBALES ------------------------

alert_count        = 0        # Nombre total d’alertes confirmées (personnes, voitures, tailgating)
detection_buffer   = {}       # Dictionnaire : label -> compteur de frames consécutives (anti-faux positifs)
TAILGATE_BUFFER    = 0        # Compteur du nombre de frames où une voiture est trop proche
LAST_POI_NOTIFIED  = {}       # Dictionnaire : nom du POI -> timestamp de dernière notification (évite le spam)
POI_CACHE          = {}       # Cache mémoire simple : stocke résultats des requêtes Google Places
alert_history      = []       # Historique du nombre d’alertes [(timestamp, alert_count)] → pour statistiques

# Chargement du modèle YOLOv8 (nano = plus rapide, moins précis → suffisant pour un prototype)
model = YOLO("yolov8n.pt")    # Télécharge auto le modèle si absent
# ------------------------------- UTILS AUDIO ---------------------------------

# Fonction pour choisir un fichier audio valide dans une liste
def choose_from_files(files, default=None):
    # files : liste de chemins vers des fichiers audio (list[str])
    # default : fichier par défaut si aucun fichier de la liste n’existe (str|None)

    # On filtre uniquement les fichiers qui existent réellement sur disque
    existing = [f for f in files if os.path.exists(f)]

    # Si aucun fichier n’est trouvé, on renvoie la valeur par défaut
    if not existing:
        return default

    # Sélection pseudo-aléatoire : index basé sur le temps actuel
    # Cela évite d’utiliser "import random" et garde une rotation naturelle
    idx = int(time.time()) % len(existing)

    # Retourne le fichier choisi
    return existing[idx]


# Fonction pour jouer un son en évitant de bloquer le programme principal
def play_sound(path):
    # path : chemin du fichier audio à jouer (str|None)

    # Si aucun fichier n’est fourni → on ne fait rien
    if not path:
        return

    # Vérifie si le fichier existe sur le disque
    if not os.path.exists(path):
        logger.warning(f"Son introuvable : {path}")  # Trace un avertissement dans les logs
        return

    # playsound() est bloquant → donc on le lance dans un thread séparé
    # Thread "daemon" = il s’arrête automatiquement quand le programme principal se termine
    threading.Thread(target=lambda: playsound(path), daemon=True).start()


# Fonction pour sélectionner un fichier audio correspondant à un type de POI
def select_poi_audio(types_list):
    # types_list : liste de types associés à un POI (ex. ["school", "hospital"])

    # On parcourt les types connus dans l’ordre défini dans POI_TYPES
    for t in POI_TYPES:
        # Si le type figure à la fois dans la liste du POI et dans notre dictionnaire
        if t in types_list and t in POI_AUDIO_MAP:
            # On choisit un fichier spécifique, sinon on renvoie le son générique
            return choose_from_files(POI_AUDIO_MAP[t], default=POI_SOUND) or POI_SOUND

    # Aucun type reconnu → on renvoie le son générique
    return POI_SOUND

# ------------------------------- UTILS GÉNÉRAUX ------------------------------

# Fonction pour consigner une alerte (log + console + historique)
def log_alert(obj_name, extra=""):
    # obj_name : str → nom de l’objet ou type d’alerte (ex. "person", "car", "tailgating")
    # extra    : str → informations complémentaires (ex. distance estimée)

    global alert_count  # On précise qu’on va modifier la variable globale alert_count

    # On incrémente le compteur global des alertes
    alert_count += 1

    # Prépare le message de l’alerte
    msg = f"Alerte {alert_count} : {obj_name} {extra}"

    # Trace le message dans le fichier log
    logger.info(msg)

    # Affiche aussi le message dans la console (pratique pour démonstration en direct)
    print(msg)

    # Ajoute un point (timestamp, alert_count) à l’historique
    # Cet historique servira pour afficher un graphique Matplotlib
    alert_history.append((time.time(), alert_count))


# Fonction pour estimer la distance à un objet en fonction de sa bounding box
def estimate_distance_from_bbox(box, frame_height):
    # box : bounding box (détection YOLO) → contient coordonnées [x1, y1, x2, y2]
    # frame_height : hauteur de l’image (pixels)

    try:
        # On essaie de récupérer les coordonnées sous forme de tableau numpy
        xy = box.xyxy[0].cpu().numpy()
    except Exception:
        # Si ça échoue, on prend directement la valeur brute
        xy = box.xyxy[0]

    # Conversion en entiers des coordonnées
    x1, y1, x2, y2 = map(int, xy)

    # Calcul de la hauteur de la boîte
    bbox_height = max(1, y2 - y1)  # Toujours ≥ 1 pour éviter division par zéro

    # Ratio : hauteur de la bbox par rapport à la hauteur totale de l’image
    height_ratio = bbox_height / max(1, frame_height)

    # Estimation heuristique de la distance (inversement proportionnelle au ratio)
    distance_m = DISTANCE_SCALE / max(height_ratio, 1e-6)

    # On borne la distance pour éviter les valeurs absurdes
    return max(MIN_DISTANCE_M, min(MAX_DISTANCE_M, distance_m))


# Fonction pour calculer la distance entre deux points GPS (formule de Haversine)
def haversine_distance_m(lat1, lon1, lat2, lon2):
    # lat1, lon1 : coordonnées GPS du premier point (float)
    # lat2, lon2 : coordonnées GPS du second point (float)
    # Retour : distance en mètres (float)

    R = 6371e3  # Rayon moyen de la Terre en mètres

    # Conversion des latitudes en radians
    phi1, phi2 = math.radians(lat1), math.radians(lat2)

    # Différences de coordonnées
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    # Formule de Haversine (calcul d’une distance orthodromique)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance finale en mètres
    return R * c

# ---------------------------------- POI ---------------------------------------

# Fonction pour récupérer les POI proches d’une position
def get_nearby_places(api_key, location, radius, types):
    # api_key  : clé API Google (str) ; si None → mode simulation
    # location : position GPS de l’utilisateur, au format "lat,lon"
    # radius   : rayon de recherche en mètres
    # types    : liste des types de lieux recherchés (list[str])

    if api_key is None:
        # MODE SIMULATION : pas d’appel API, on génère des POI fictifs
        lat, lon = map(float, location.split(","))  # On sépare "lat,lon" en 2 floats

        # On renvoie 3 POI simulés avec des coordonnées légèrement modifiées
        return [
            {"name": "École Simone",  "geometry": {"location": {"lat": lat + 0.001,  "lng": lon + 0.001}},  "types": ["school"]},
            {"name": "Hôpital Saint", "geometry": {"location": {"lat": lat + 0.002,  "lng": lon - 0.0005}}, "types": ["hospital"]},
            {"name": "Stade Central", "geometry": {"location": {"lat": lat - 0.0015, "lng": lon + 0.002}},  "types": ["stadium"]},
        ]

    # Si API activée → on prépare une clé de cache pour éviter appels répétés
    key = (location, tuple(types))

    # Si on a déjà les données en cache et qu’elles datent de moins de 60s → on les réutilise
    if key in POI_CACHE and (time.time() - POI_CACHE[key]["ts"]) < 60:
        return POI_CACHE[key]["data"]

    # Construction de l’URL d’appel à Google Places API Nearby Search
    url = (
        "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        f"?location={location}&radius={radius}&types={'|'.join(types)}&key={api_key}"
    )

    try:
        # On lance la requête avec un timeout court pour éviter de bloquer le programme
        r = requests.get(url, timeout=5)

        # On récupère les résultats
        data = r.json().get("results", [])

        # On les stocke dans le cache
        POI_CACHE[key] = {"data": data, "ts": time.time()}

        return data

    except Exception as e:
        # Si une erreur survient (réseau, API, etc.), on trace dans les logs
        logger.error(f"Erreur API Google Places : {e}")
        # On retourne une liste vide pour éviter un crash
        return []


# Fonction pour notifier si on est proche d’un POI
def notify_poi_if_close(user_location_str, api_key):
    # user_location_str : str → position de l’utilisateur ("lat,lon")
    # api_key           : str|None → clé API ou None si mode simulation

    # On convertit la position utilisateur en floats
    u_lat, u_lon = map(float, user_location_str.split(","))

    # On récupère la liste des POI proches
    places = get_nearby_places(api_key, user_location_str, POI_SEARCH_RADIUS_METERS, POI_TYPES)

    # On boucle sur tous les lieux trouvés
    for p in places:
        # On récupère le nom du lieu (ou "Lieu" par défaut)
        name  = p.get("name", "Lieu")

        # On récupère la latitude et longitude du POI
        p_lat = p.get("geometry", {}).get("location", {}).get("lat")
        p_lon = p.get("geometry", {}).get("location", {}).get("lng")

        # Si les coordonnées sont manquantes, on ignore ce POI
        if p_lat is None or p_lon is None:
            continue

        # On calcule la distance utilisateur ↔ POI
        dist_m = haversine_distance_m(u_lat, u_lon, p_lat, p_lon)

        # On récupère la liste des types associés au POI
        poi_types = p.get("types", [])

        # On crée un identifiant basique pour ce POI (ici, juste son nom)
        poi_id = f"{name}"

        # Timestamp actuel
        now_ts = time.time()

        # Condition : POI à moins de 200 m ET pas notifié depuis 30s
        if dist_m <= POI_NOTIFY_DISTANCE_METERS and (now_ts - LAST_POI_NOTIFIED.get(poi_id, 0)) > 30:
            # Prépare le message d’alerte
            msg = f"⚠️ Approche : {name} à ~{int(dist_m)} m"

            # Affiche le message dans la console
            print(msg)

            # Trace aussi dans les logs
            logger.info(msg)

            # Choisit un son spécifique selon le type du POI (ou générique sinon)
            sound_to_play = select_poi_audio(poi_types)

            # Joue le son (thread non bloquant)
            play_sound(sound_to_play)

            # Enregistre le timestamp de notification pour ce POI
            LAST_POI_NOTIFIED[poi_id] = now_ts
# ------------------------------ NOTIFICATIONS --------------------------------

# Fonction qui envoie un message d’apaisement si le conducteur colle trop une voiture
def calm_driver_notify():
    # Cette fonction est appelée lorsqu’un "tailgating" est détecté de manière persistante.

    # Liste de messages bienveillants et rassurants (affichés console + logs)
    messages = [
        "Ralentissez et respirez profondément… la route est plus sûre quand on garde ses distances.",
        "La sécurité n’est jamais une perte de temps… mieux vaut arriver un peu plus tard que de ne jamais arriver.",
        "Conduire, c’est partager la route. Respecter l’espace des autres, c’est aussi se protéger soi-même.",
        "Un instant de patience peut éviter un accident. Gardez vos distances, votre voyage n’en sera que plus serein.",
        "Rappelez-vous : un conducteur calme protège sa vie et celle des autres. La route n’est pas une course.",
    ]

    # On choisit un message pseudo-aléatoirement en se basant sur le temps
    msg = messages[int(time.time()) % len(messages)]

    # On trace le message dans les logs (niveau INFO)
    logger.info(f"Message calmant : {msg}")

    # On affiche aussi dans la console pour le retour visuel
    print(f"💬 {msg}")

    # On choisit un fichier audio calmX.mp3 existant, ou calm_alert.mp3 en fallback
    sound_to_play = choose_from_files(CALM_SOUNDS, default=CALM_SOUND)

    # On joue le son choisi dans un thread séparé
    play_sound(sound_to_play)


# ---------------------------- SIMULATION GPS ---------------------------------

# Générateur qui produit des positions GPS fictives pour tester le système sans capteur réel
def simulated_position_generator():
    # Point de base choisi : Alger centre (latitude, longitude)
    base_lat, base_lon = 36.752778, 3.042222

    # Compteur d’étapes (sert à faire varier la position avec sin/cos)
    step = 0

    # Boucle infinie (ce générateur ne s’arrête jamais)
    while True:
        # On simule un petit déplacement en latitude avec une fonction sinusoïdale
        lat = base_lat + 0.0001 * math.sin(step / 10.0)

        # On simule un petit déplacement en longitude avec une fonction cosinusoïdale
        lon = base_lon + 0.00015 * math.cos(step / 10.0)

        # On incrémente le compteur à chaque appel
        step += 1

        # On renvoie la position GPS sous forme de chaîne "lat,lon"
        yield f"{lat},{lon}"


# ------------------------------ STATISTIQUES ---------------------------------

# Fonction qui affiche en temps réel l’évolution du nombre d’alertes
def show_statistics():
    # Cette fonction tourne dans un thread dédié pour ne pas bloquer la détection vidéo.

    # Activation du mode interactif de Matplotlib (rafraîchissement continu)
    plt.ion()

    # Création d’une figure et d’un axe pour le graphique
    fig, ax = plt.subplots()

    # Boucle infinie : le graphique se mettra à jour tant que le programme tourne
    while True:
        # On vérifie s’il existe au moins une alerte dans l’historique
        if alert_history:
            # On prend le premier timestamp comme référence (t0)
            t0 = alert_history[0][0]

            # Liste des temps relatifs (secondes depuis la 1ʳᵉ alerte)
            times = [t - t0 for t, _ in alert_history]

            # Liste des valeurs du compteur d’alertes
            counts = [c for _, c in alert_history]

            # On nettoie l’axe pour redessiner un graphique propre à chaque boucle
            ax.clear()

            # On trace la courbe du nombre d’alertes dans le temps
            ax.plot(times, counts, marker="o", color="red")

            # On ajoute un titre et des labels aux axes
            ax.set_title("Évolution des alertes dans le temps")
            ax.set_xlabel("Temps écoulé (s)")
            ax.set_ylabel("Nombre d'alertes")

            # On ajoute une grille légère pour faciliter la lecture
            ax.grid(True, linestyle="--", alpha=0.5)

        # On fait une pause d’1 seconde avant la prochaine mise à jour
        plt.pause(1)
# ---------------------------------- MAIN -------------------------------------

# Fonction principale : c’est ici que tout le système s’exécute
def main():
    # On ouvre la caméra par défaut (index 0 = webcam intégrée ou première caméra USB)
    cap = cv2.VideoCapture(0)

    # Vérifie si la caméra s’est bien ouverte
    if not cap.isOpened():
        # Si erreur : on envoie un message dans les logs
        logger.error("Impossible d'ouvrir la caméra")
        # Et on quitte proprement la fonction
        return

    # Générateur de positions GPS simulées (lat,lon)
    pos_gen = simulated_position_generator()

    # Intervalle entre deux vérifications de POI (en secondes)
    poi_check_interval = 5.0

    # Timestamp de la dernière vérification POI
    last_poi_check = 0.0

    # On précise qu’on va modifier la variable globale TAILGATE_BUFFER
    global TAILGATE_BUFFER

    # Message d’accueil affiché dans la console
    print("🚀 Détection démarrée. Appuyez sur 'q' pour quitter.")

    # On lance le thread pour afficher les statistiques en direct
    threading.Thread(target=show_statistics, daemon=True).start()

    # ---------------- BOUCLE PRINCIPALE ----------------
    while True:
        # On lit une image (frame) depuis la caméra
        ret, frame = cap.read()

        # Si la lecture échoue (ex. caméra débranchée)
        if not ret:
            # On trace l’erreur dans les logs
            logger.error("Lecture frame impossible")
            # Et on sort de la boucle
            break

        # On récupère la taille de l’image (hauteur, largeur)
        frame_h, frame_w = frame.shape[:2]

        # On applique YOLOv8 sur l’image (imgsz fixe pour vitesse et stabilité)
        results = model(frame, imgsz=640, verbose=False)

        # On génère une copie annotée de l’image (boîtes + labels dessinés)
        annotated = results[0].plot()

        # Ensemble des labels vus dans la frame courante
        labels_this_frame = set()

        # On parcourt les résultats (batch YOLO)
        for r in results:
            # On parcourt chaque bounding box détectée
            for box in r.boxes:
                # Classe prédite (entier)
                cls = int(box.cls[0])
                # Nom lisible de la classe (ex. "person", "car")
                label = model.names[cls]

                # On ajoute ce label à l’ensemble des labels de cette frame
                labels_this_frame.add(label)

                # On incrémente le compteur d’apparition consécutive de ce label
                detection_buffer[label] = detection_buffer.get(label, 0) + 1

                # -------------- Cas particulier : voiture (détection de tailgating) --------------
                if label == "car":
                    # On estime la distance à la voiture détectée
                    dist_m = estimate_distance_from_bbox(box, frame_h)

                    # Coordonnées de la bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Centre de la bounding box (abscisse)
                    cx = (x1 + x2) / 2

                    # Seuil pour estimer si la voiture est dans "notre voie"
                    lane_center_threshold = frame_w * 0.25

                    # Si distance trop proche ET voiture bien alignée
                    if dist_m <= TAILGATE_THRESHOLD_METERS and abs(cx - frame_w / 2) <= lane_center_threshold:
                        # On incrémente le buffer de tailgating
                        TAILGATE_BUFFER += 1
                    else:
                        # Sinon, on réduit progressivement le buffer (anti faux positifs)
                        TAILGATE_BUFFER = max(0, TAILGATE_BUFFER - 1)

                    # Si le buffer dépasse le seuil → on notifie un "calm driver"
                    if TAILGATE_BUFFER >= TAILGATE_FRAMES_REQUIRED:
                        # On envoie le message + son calmant
                        calm_driver_notify()
                        # On sauvegarde une capture d’image
                        cv2.imwrite(f"{CAPTURE_DIR}/tailgate_{int(time.time())}.jpg", frame)
                        # On consigne l’événement dans les logs
                        log_alert("tailgating", f"(dist={int(dist_m)}m)")
                        # On remet le buffer à zéro pour éviter le spam
                        TAILGATE_BUFFER = 0

                # -------------- Cas général : personnes et voitures persistantes --------------
                if label in ["person", "car"] and detection_buffer.get(label, 0) >= ALERT_THRESHOLD_FRAMES:
                    # On consigne l’alerte dans les logs + historique
                    log_alert(label)
                    # On joue le son d’alerte (dans un thread séparé)
                    play_sound(ALERT_SOUND)
                    # On sauvegarde une capture image
                    cv2.imwrite(f"{CAPTURE_DIR}/alert_{alert_count}_{label}.jpg", frame)
                    # On réinitialise le buffer de ce label
                    detection_buffer[label] = 0

        # ---------------- Nettoyage des buffers ----------------
        # Pour chaque label non vu dans cette frame → on décrémente son compteur
        for lbl in list(detection_buffer.keys()):
            if lbl not in labels_this_frame:
                detection_buffer[lbl] = max(0, detection_buffer[lbl] - 1)

        # ---------------- HUD (overlay visuel) ----------------
        # On affiche le nombre total d’alertes en haut à gauche
        cv2.putText(annotated, f"Alertes: {alert_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Si le buffer tailgating est actif, on affiche un avertissement
        if TAILGATE_BUFFER > 0:
            cv2.putText(annotated, "TAILGATE WARNING", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)

        # ---------------- Vérification périodique des POI ----------------
        now_ts = time.time()  # Timestamp actuel

        # Si assez de temps est passé depuis la dernière vérification
        if now_ts - last_poi_check > poi_check_interval:
            # On met à jour la dernière vérification
            last_poi_check = now_ts
            # On récupère une nouvelle position GPS simulée
            user_loc = next(pos_gen)
            # On lance la vérification POI dans un thread séparé
            threading.Thread(
                target=notify_poi_if_close,
                args=(user_loc, GOOGLE_API_KEY),
                daemon=True,
            ).start()

        # ---------------- Affichage du flux vidéo ----------------
        # On montre la frame annotée dans une fenêtre OpenCV
        cv2.imshow("Sécurité Routière - Amélioré (Prototype)", annotated)

        # Si l’utilisateur appuie sur la touche 'q' → on quitte la boucle
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # ---------------- Libération des ressources ----------------
    # On libère la caméra (indispensable pour ne pas la bloquer après exécution)
    cap.release()

    # On ferme toutes les fenêtres OpenCV ouvertes
    cv2.destroyAllWindows()


# ----------------------------- LANCEUR DE SCRIPT ------------------------------

# Point d’entrée standard du script Python
if __name__ == "__main__":
    # On appelle la fonction main()
    # → c’est ici que démarre l’exécution du programme
    main()
