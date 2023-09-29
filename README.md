# Visualisation pour les concerts Retrópico


## Installation
Installer les paquets Python 3 dans le fichier `requirements.txt`

## Utilisation
1. Brancher une carte son Scarlett Solo USB avec le piano sur l'entrée 1 et le pad sur l'entrée 2
2. Démarrer l'application en éxécutant dans le terminal la commande `python spectrogramme.py`


## Logique du programme
### Piste du piano
1. **Acquisition du signal** : avec la librairie pyaudio le signal est échantillonné par blocs de nombres. Un nombre sur deux provient alternativement de l'entrée 1 puis de l'entrée 2 de la carte.
2. **Transformée de Fourrier du son du piano** : on effectue plusieurs fois (4 fois) la transformée de fourrier du son en utilisant une fenetre glissante plus ou moins large en temps, pour capturer différentes plages de fréquences avec une bonne réactivité. La fenêtre la plus courte est utilisée pour les hautes fréquences, qui s'affichent donc rapidement sur l'écran. Pour les basses fréquences en revanche, on utilise une fenêtre plus longue qui permet d'avoir une meilleure résolution en fréquence au prix d'une réactivité plus faible.
3. **Discrétisation sur une grille logarithmique** : les fonctions FFT donnent des amplitudes en fonction de fréquences qui sont échantillonées de manière linéaires. Comme on veut afficher les fréquences sur une échelle de fréquence logarithmique, on rééchantillonne les amplitudes sur une nouvelle grille qui correspond à l'image que l'on veut afficher plus tard.
4. **Transformation en image** : Tout un mic mac avec des couleurs etc je te laisse découvrir

### Piste batterie
1. Transformation du signal en enveloppe d'amplitude
2. Convolution de l'enveloppe avec des cercles concentriques
3. Multiplication à l'image du piano

## Exemple

https://github.com/gaspardbenoit/retropico-visualisation/assets/142936687/9633a3e3-d7cd-4ad8-b54b-8090ece5e330


