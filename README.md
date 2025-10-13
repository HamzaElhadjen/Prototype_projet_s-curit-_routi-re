# Cockpit Intelligent pour la Sécurité Routière
Application de l’IA aux systèmes embarqués

Projet d’étude réalisé dans le cadre de ma candidature en Master 1 Intelligence Artificielle et Systèmes Embarqués (septembre 2025).

---

## Objectif
Concevoir un cockpit intelligent capable de :
- Détecter en temps réel les obstacles (voitures, piétons)
- Assister le conducteur via des signaux visuels et sonores
- Fournir un tableau de bord interactif (jauges, alertes, carte)

---

## Architecture du projet
Le projet est développé en Python et se compose de deux modules principaux :

- **main.py** : interface graphique basée sur PyQt5
  - Affiche la vidéo annotée
  - Gère les jauges et statistiques
  - Propose un mode Jour/Nuit
- **core.py** : moteur de détection basé sur YOLOv8
  - Détection des objets (voitures, piétons)
  - Estimation de distance
  - Gestion des alertes (visuelles et sonores)
  - Simulation GPS et Points d’intérêt (POI)

La communication est assurée de manière asynchrone via QThread et le mécanisme de signaux/slots.

---

## Fonctionnalités principales
- Détection d’objets (YOLOv8)  
- Estimation de distance et prévention du non-respect des distances de sécurité  
- Alertes sonores et historique des alertes  
- Tableau de bord interactif : vitesse simulée, état de l’IA, voyant obstacle  
- Carte des points d’intérêt (écoles, hôpitaux, stades)  
- Mode Jour/Nuit et statistiques en temps réel  

---

## Perspectives d’évolution
- Déploiement embarqué sur Raspberry Pi ou NVIDIA Jetson Nano  
- Ajout de capteurs avancés (LiDAR, radar, caméras stéréo)  
- Intégration de Google Maps API pour un suivi GPS en temps réel  
- Détection de la fatigue et de la distraction du conducteur via IA  
- Analyse santé connectée (par exemple fréquence cardiaque via Apple Watch)  
- Notifications multimodales : vibrations, signaux lumineux, sons  
- Extension vers des fonctionnalités d’assistance semi-autonome  

----

## Technologies utilisées
- Python 3
- YOLOv8 (Ultralytics)
- PyQt5 (interface graphique)
- Threads (QThread)

--

## Auteur
- Elhadjen Hamza Hocine

---

