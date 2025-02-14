# Bildverarbeitungssystem mit OpenCV und Tiefenkamera

In diesem Projekt habe ich ein Bildverarbeitungssystem programmiert, das OpenCV und eine Tiefenkamera (Stereokamera) nutzt. Dabei werden Tiefeninformationen in Echtzeit verarbeitet, um spannende Features wie Bildbinarisierung und Objekterkennung umzusetzen.

## Features

- **Bildbinarisierung:**  
  Das Tiefenbild wird normalisiert und binarisiert, um wichtige Bereiche hervorzuheben.

- **Objekterkennung und Umrahmung:**  
  Erkanntes Objekt (z. B. eine Hand) wird automatisch detektiert und mit einem roten Rechteck umrahmt.

- **Histogramm-basierte Schwellenwertbestimmung:**  
  Dynamische Anpassung des Schwellenwerts basierend auf dem Tiefenbild-Histogramm.

- **Echtzeitverarbeitung:**  
  Unterstützung für Live-Video, Videoaufnahmen und spätere Auswertungen.