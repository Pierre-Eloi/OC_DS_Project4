# Projet 4 : Anticiper les besoins en consommation électrique de bâtiments
*Pierre-Eloi Ragetly*

Ce projet fait parti du parcours DataScientist d'Open Classrooms.

L'objectif principal est de prédire la consommation totale d'énergie ainsi que les émissions de CO2 des bâtiments non destinés à l'habitation de la ville de Seattle.  
Les données utilisées proviennent du projet kaggle [*SEA Building Energy Benchmarking*](https://www.kaggle.com/city-of-seattle/sea-building-energy-benchmarking#2015-building-energy-benchmarking.csv).

L'analyse a été découpée en trois notebooks:
- **Project4_Data_Wrangling** qui regroupe toutes les opérations nécessaires au nettoyage des données
- **Project4_Feature_Engineering** qui regroupe les opérations d'ingénierie de variables
- **Project4_Data_Modeling** qui regroupe les résultats des différents modèles utilisés

Les librairies python nécessaires pour pouvoir lancer le notebook sont regroupées dans le fichier *requirements.txt*

Toutes les fonctions créées afin de mener à bien le projet ont été regroupées dans le dossier **functions**.
- les fonctions de nettoyage (incluant un pipeline à la fin) dans *cleaning.py*
- les fonctions de traitement des variables (là encore un pipeline a été créé) dans *feat_engineering.py*
- les fonctions permettant d'optimiser les différents hyperparamètres de chaque modèle dans *ml_modeling.py*

Durant l'étude, les principaux modèles de regression disponibles en machine learning ont été testés. Le tableau ci-dessous regroupe les résultats pour la modélisation des émissions en CO2.

![](/charts)
