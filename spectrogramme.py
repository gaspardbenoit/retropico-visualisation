
#%% 

import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import cv2
import pyaudio
import colorsys
import os

# import sys
import time

# PARAMETRES

# f = 2 ** n/12
# f ** 12 = 2 ** n
# n = ln(f**12)
# a0 55Hz

couleur = True

notes = ['A','Bb','B','C','C#','D','D#','E','F','Gb','G','Ab']

couleurs = ['#F3A1BE','#95E95A','#F8A986','#ABBFF3','#EFBF2B','#E498FC','#BFD434','#FAA39B','#78D7FF','#F2B85F','#CCAEFE','#D6D300']

couleurs_foncees = '#B25375','#3C9400','#B85226','#3568EF','#A97F01','#9C49B6','#8DA300','#CD6258','#018EC9','#A76300','#8E68CD','#929113'
couleurs_bgr = [tuple(int(h[i+1:i+3], 16) for i in (4, 2, 0)) for h in couleurs_foncees]


# frequences_notes = [(np.log10(55*2**(i/12)), notes[i%12] ) for i in range(0,5*12)]

# frequences_notes





from screeninfo import get_monitors
for m in get_monitors():
    largeur_dernier_moniteur = m.width
    hauteur_dernier_moniteur = m.height
    ratio_ecran = largeur_dernier_moniteur / hauteur_dernier_moniteur
    print(str(m))


largeur = 800
hauteur = int(np.round( largeur / ratio_ecran ))
print('Hauteur de l\'image :',hauteur,'pixels')


# On précalcule les anneaux concentriques
distances = np.ndarray.astype(np.zeros((largeur,hauteur)),dtype=int)

X,Y = np.ix_(np.arange(largeur),np.arange(hauteur))
vitesse_cercles = 10
distances = np.round((np.sqrt((X-largeur/2)**2 + (Y-hauteur/2)**2)) / vitesse_cercles,0)
# print(distances)
distances = distances.flatten().astype(int)



# Initialisation de pyaudio
FORMAT = pyaudio.paInt16
CHANNELS = 2
echantillonage = 44100 # echantillons par secondes
echantillons_par_tampon = 2048 # commande la fréquence de rafraichissement qu'on veut a 20Hz


# la memoire qui va contenir le son du piano pendant les 10 dernières secondes pour le spectre
memoire = np.zeros(10*echantillonage)
# la mémoire qui va contenir le volume moyen du son du pad a chaque itération
enveloppe = np.zeros(10*echantillonage)

temps_derniere_mesure = time.time()

nombre_de_pixels_frequences = 720

spectre = np.zeros(nombre_de_pixels_frequences)
phase = np.zeros(nombre_de_pixels_frequences)



p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')

identifiant_carte_son = 0
for i in range(0, numdevices):
    if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))
        if p.get_device_info_by_host_api_device_index(0, i).get('name') == 'Scarlett Solo USB' :
            identifiant_carte_son = i



stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=echantillonage,
    input_device_index=identifiant_carte_son, #mettre ici la source de son désirée
    input=True,
    # output=True,
    frames_per_buffer=echantillons_par_tampon,
)

# exit()

# 2**(1/12) # = 1.059 (6% entre chaque note)


facteur_lissage = 0.2
# un facteur de lissage entre chaque itération de spectre (lissage vertical sur l'image)


# 44100, 4410 : 5kHz a l'index 1000
# 44100, 22050 : 1kHz a l'index 1000
# 44100, 44100 : 300Hz a l'index 1000

# la fonction qui a l'index n associe la frequence vaut :
# f(1000) = 1000 / 2 * echantillonage / longueur_fenetre_spectre

# par ailleurs la frequence la plus haute est egale a f(fenetre/2) = echantillonage / 4






division_octaves = 96 # 1.45% entre chaque division
# le nombre de pixels pour chaque octave


frequences_notes = [55*2**(i/division_octaves) for i in range(0,8*division_octaves) if 55*2**(i/division_octaves) < 10000]
# les frequences associées a chaque pixel

len(frequences_notes)



# IMAGE
# on précalcule les couleurs

image = np.zeros(shape=(nombre_de_pixels_frequences,300)) # l'axe temporel en premier, frequentiel ensuite
image_couleur_reference = np.zeros(shape=(nombre_de_pixels_frequences,300,3)) # l'axe temporel en premier, frequentiel ensuite
for i in range(0,nombre_de_pixels_frequences):
    for j in range(0,300):
        couleur_boucle = couleurs_bgr[int(round((12*i/division_octaves)))%12]
        hsv = colorsys.rgb_to_hsv(couleur_boucle[2]/255,couleur_boucle[1]/255,couleur_boucle[0]/255)

        # les couleurs changent a chaque ligne
        hsv = ( (hsv[0]+j/300) % 1 , hsv[1], hsv[2])
        rgb = np.asarray( colorsys.hsv_to_rgb(*hsv) )
        bgr = np.flip(np.round(rgb*255,0))
        image_couleur_reference[i,j,:] = bgr
image_couleur = np.zeros(shape=(nombre_de_pixels_frequences,300,3)) # np.asarray(image_couleur_reference,dtype='int')
image 
item_image = pg.ImageItem( image=image_couleur_reference, levels=(0,255) ) # create example image
# conteneur_image.addItem( item_image )

#%%

# initialisation de la fenêtre dans laquelle sera affichée l'image
cv2.namedWindow("frame", cv2.WINDOW_AUTOSIZE)
# cv2.namedWindow("carre", cv2.WINDOW_AUTOSIZE)



def re_echantilloner(arr, n):
    # n est le nombre d'echantillons moyennés dans chaque groupe
    end =  n * int(len(arr)/n)
    return np.mean(arr[:end].reshape(-1, n), 1)



# une fonction qui prend en entree : 
# - du son
# - une résolution voulue en fréquence
# - 
# Qui renvoie : un spectre, les fréquences associées

# dictionnaire_precalcul = {}

def bout_de_spectre(son,resolution_hertz,sous_echantillonage):
    debut = time.time()
    longueur_fenetre_spectre = int(echantillonage / resolution_hertz)
    # print("Durée de l\'échantillon : ",longueur_fenetre_spectre/echantillonage, 'secondes')

    # frequences_sortie_fft = np.linspace(0, int(echantillonage/sous_echantillonage / 2),int(echantillonage/2/resolution_hertz/sous_echantillonage))

    son_pour_spectre = re_echantilloner(son[-longueur_fenetre_spectre:], sous_echantillonage)
    # print('len(son_opour_spectre)',len(son_pour_spectre))

    fenetre_lissage_spectre = np.blackman(len(son_pour_spectre))

    nouveau_spectre = np.fft.fft(
        fenetre_lissage_spectre * son_pour_spectre
    )
    frequences_sortie_fft = np.fft.fftfreq(len(son_pour_spectre), d=sous_echantillonage/echantillonage)[0:int(len(nouveau_spectre) / 2)]
    # print('frequences sortie fft',frequences_sortie_fft)
    spectre_amplitude = np.abs(nouveau_spectre[0:int(len(nouveau_spectre) / 2)])
    # * 2 / ( longueur_fenetre_spectre)
    # seule la moitié nous intéresse, la deuxième est identique
    spectre_phase = np.angle(nouveau_spectre[0:int(len(nouveau_spectre) / 2)])
    decalage = frequences_sortie_fft*2*np.pi*temps_derniere_mesure
    spectre_phase = (spectre_phase - decalage) % (2*np.pi) - np.pi

    fin = time.time()
    print('bout de spectre',np.int(np.round((fin-debut)*1000)),'ms avec resolution_hertz', np.round(resolution_hertz,1),' et secondes',np.round(len(son_pour_spectre)/echantillonage,3))
    return spectre_amplitude / np.sqrt(len(son_pour_spectre)), frequences_sortie_fft, spectre_phase



def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def interpoler(y):
    nans, x= nan_helper(y)
    y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    return y

def grand(tableau):
    if len(tableau) > 0:
        return np.max(tableau)
    return np.nan
leplusgrand = np.vectorize(grand)

# le mot quantifié fait référence a la "quantization" sur les logiciels de musiques. C'est à dire que cette fonction va rééchantilloner sur une grille logarithmique les données de la transformée de fourrier qui sont sur une linéaire en fréquence
def spectre_quantifie(
        son,
        sous_echantillonage,
        frequence_min,
        frequence_max
    ):
    debut = time.time()

    frequence_max_arrondie = 55*2**(round(np.log2((max(frequence_max,1)/55))*division_octaves)/division_octaves)
    frequence_min_arrondie = 55*2**(round(np.log2((max(frequence_min,1)/55))*division_octaves)/division_octaves)

    numero_min = round(np.log2((max(frequence_min,1)/55))*division_octaves)
    numero_max = round(np.log2((max(frequence_max,1)/55))*division_octaves)
    # print('Numéros de pixels :',numero_min,numero_max)

    # print("Frequences arrondies :",frequence_min_arrondie,frequence_max_arrondie)


    resolution_hertz = frequence_min *( 2**(1/division_octaves) - 1 ) * 2
    # print('Résolution :',resolution_hertz,'hertz')
    # print('Sous echantillonage :',echantillonage / 2 / frequence_max,'frequence max :',frequence_max)
    sous_echantillonage = np.maximum(int(np.floor(echantillonage / 2 / frequence_max / 2)), 1)
    # sous_echantillonage = 1
    

    spectre_amplitude, frequences_sortie_fft, spectre_phase = bout_de_spectre(son,resolution_hertz,sous_echantillonage)
    # print('Frequences sortie fft :',np.min(frequences_sortie_fft),np.max(frequences_sortie_fft))

    # on enleve les bornes inf et sup
    spectre_filtre = [p for i, p in enumerate(spectre_amplitude) if frequences_sortie_fft[i] < frequence_max_arrondie and frequences_sortie_fft[i] >= frequence_min_arrondie]

    phases_filtrees = [p for i, p in enumerate(spectre_phase) if frequences_sortie_fft[i] < frequence_max_arrondie and frequences_sortie_fft[i] >= frequence_min_arrondie]

    frequences_filtrees = [f for f in frequences_sortie_fft if f < frequence_max_arrondie and f >= frequence_min_arrondie]

    # on fait l'intégrale sur la grille logarithmique

    numeros_frequence = [ max(round(np.log2((max(f,1)/55))*division_octaves),0) for f in frequences_filtrees ]


    frequences_pixels = [f for f in frequences_notes if f < frequence_max_arrondie and f >= frequence_min_arrondie]

    # print('Fréquences pixels :',len(frequences_pixels),'Fréquences filtrées :',len(frequences_filtrees))
    # print('len(np.unique(numeros_frequence))',len(np.unique(numeros_frequence)))
    # print('len(numeros_frequence), len(spectre_filtre)',len(numeros_frequence),len(spectre_filtre))

    integrale_phase = np.bincount(numeros_frequence,phases_filtrees)
    intervalles = np.bincount(numeros_frequence)
    intervalles[intervalles==0] = 1
    # il y a des intervalles avec des zeros (resolution_hertz trop faible)

    # alternative à l'intégrale : on prend le maximum de chaque intervalle

    pixels_uniques = np.unique(numeros_frequence, return_index=True)
    # print('pixels',numeros_frequence,pixels_uniques)
    volumes = leplusgrand(np.split(spectre_filtre, pixels_uniques[1][1:]))

    numeros_pixels = np.arange(numero_min,numero_max+1)
    volumes_pixels = np.full(numero_max-numero_min+1,np.nan)
    # print('numeros_pixels',numeros_pixels,pixels_uniques)
    volumes_pixels[np.isin(numeros_pixels,pixels_uniques[0])] = volumes

    # volumes_pixels = interpoler(volumes_pixels)
    # print('volumes pixels',volumes_pixels)


    # volumes = integrale[numero_min:(numero_max+1)] / intervalles[numero_min:(numero_max+1)]
    phases = integrale_phase[numero_min:(numero_max+1)] / intervalles[numero_min:(numero_max+1)]
    # print(len(volumes[numero_min:(numero_max+1)]), len(frequences_pixels),len(np.unique(numeros_frequence)))
    fin = time.time()
    print('spectre quantifié',np.int(np.round((fin-debut)*1000)),'ms avec frequence max',np.round(frequence_max),'et sous echantillonage',sous_echantillonage)
    # print('Fréquence min et max :',frequence_min,frequence_max)
    # print('Fréquences pixels :',len(frequences_pixels),'Fréquences filtrées :',len(frequences_filtrees),'Volumes pixels :',len(volumes_pixels),'Phases :',len(phases))

    volumes = volumes[0:len(frequences_pixels)]
    phases = phases[0:len(frequences_pixels)]
    # phases = np.concatenate([phases,np.zeros(len(frequences_pixels)-len(phases))])

    return volumes_pixels[:-1], frequences_pixels, phases
    # on interpole les valeurs nulles


# cette fonction effectue plusieurs fois la transformée de fourrier du son sur différentes fenêtres et différents sous-échantillonages. Le but est d'avoir une meilleur résolution spectrale dans les basses fréquences ainsi qu'une plus faible latence dans les aigus.
def spectre_complet(son):
    debut = time.time()

    sous_echantillonage = 1
    n = 4
    # diviser le domaine 55-10k en n bouts logarythmiques
    bornes = [ 55 * np.exp(i * (np.log(10000)-np.log(55))/n) for i in np.arange(n+1) ]
    # print(bornes)

    donnees = [spectre_quantifie(son,sous_echantillonage,bornes[i],bornes[i+1]) for i in np.arange(n)]
    # print(len(donnees))

    volumes = np.concatenate([segment[0] for segment in donnees]) / 10
    frequences_pixels = np.concatenate([segment[1] for segment in donnees])
    nouvelles_phases = np.concatenate([segment[2] for segment in donnees])
    
    # print('Pixels interpolés :',round(len(volumes[np.isnan(volumes)]) / len(volumes) * 100),'%')
    volumes = interpoler(volumes)
    # volumes_1, frequences_pixels_1, phases_1 = spectre_quantifie(son,sous_echantillonage,55,200)

    # volumes_2, frequences_pixels_2, phases_2 = spectre_quantifie(son,sous_echantillonage,200,10000)
    # volumes = np.concatenate((volumes_1, volumes_2)) / 10
    # par mesure empirique l'amplitude maximum est de 10

    # phases = np.concatenate((phases_1, phases_2))
    # frequences_pixels = np.concatenate((frequences_pixels_1, frequences_pixels_2))
    fin = time.time()
    print('spectre complet',np.int(np.round((fin-debut)*1000)),'ms')
    
    return volumes, frequences_pixels, nouvelles_phases

carre_blanc = np.ones(shape=(10,10))



# un index pour se rappeler a quelle ligne de couleur on en est
index_couleur = 0


# la fonction qui prend en entrée des rangées de pixels noir et blanc et qui les assemble en une image en couleur
def tracer(schema, data_x, data_y):
    global image, image_couleur, enveloppe, index_couleur

        
    # if schema == 'schema_onde':
    #     traces[schema] = schema_onde.plot(pen='c', width=3)
    #     schema_onde.setYRange(-1, 1, padding=0)
    #     schema_onde.setXRange(0, 2 * echantillons_par_tampon, padding=0.005)

    # if schema == 'schema_spectre':
    #     traces[schema] = schema_spectre.plot(pen='m', width=3)
    #     schema_spectre.setLogMode(x=True, y=True)
    #     schema_spectre.setYRange(-6, 0, padding=0)
    #     schema_spectre.setXRange(
    #         np.log10(100), np.log10(5000), padding=0.005)
        


    debut = time.time()
    

    data_y = data_y[0:nombre_de_pixels_frequences]

    if image.shape[0] < len(data_y):
        image = np.zeros(shape=(len(data_y),300))
        
    image = np.roll(image,-1,axis=1)
    image[:,-1] = data_y # np.log(data_y+1)
        
    image_couleur = np.roll(image_couleur,-1,axis=1)


    image_couleur[:,-1,0] = np.clip(np.multiply(image_couleur_reference[0:nombre_de_pixels_frequences,index_couleur%300,0], data_y)*100,0,255)
    image_couleur[:,-1,1] = np.clip(np.multiply(image_couleur_reference[0:nombre_de_pixels_frequences,index_couleur%300,1], data_y)*100,0,255)
    image_couleur[:,-1,2] = np.clip(np.multiply(image_couleur_reference[0:nombre_de_pixels_frequences,index_couleur%300,2], data_y)*100,0,255)

    index_couleur += 1



    # limite = np.percentile(image,99.8)
    # image_normee = image.copy()*100
    # print(np.median(image_normee),np.max(image_normee))

    # image_normee_log = np.clip((np.log(interpoler(image.copy()))/10 + 1)*2-0.1, 0,1)
    # je veux une image en echelle log avec le plancher 10-4 a zero et le max a 1 ou 0.1
    
            
    # maximum = np.max(image_normee)
    # if maximum > 0 :
    #     image_normee = image_normee / maximum

    # image_couleur[:,:,0] = np.asarray(
    #     np.multiply(image_normee,image_couleur_reference[:,0:len(data_y),0]),
    #     dtype = 'int'
    # )
    # image_couleur[:,:,1] = np.asarray(
    #     np.multiply(image_couleur_reference[:,0:len(data_y),1],image_normee),
    #     dtype = 'int'
    # )
    # image_couleur[:,:,2] = np.asarray(
    #     np.multiply(image_couleur_reference[:,0:len(data_y),2],image_normee),
    #     dtype = 'int'
    # )
    # image_couleur = np.clip(image_couleur,0,255)
    debut = time.time()

    
    # image_couleur = image_couleur / np.max(image) * 255

    # print(np.max(image_couleur),np.min(image_couleur),np.max(image),np.min(image))
    # print(np.percentile(image,90))

    cercles = enveloppe[distances].reshape((hauteur,largeur),order='F')
    cercles = cv2.merge((cercles,cercles,cercles))

    if couleur == True:
        # item_image.setImage(image_couleur,autoLevels=False)
        # print('Etape X: afficher l\'image en couleur')
        image_redimensionnee = cv2.resize(cv2.rotate(image_couleur/255.0,cv2.ROTATE_90_COUNTERCLOCKWISE),(800,500))
        # print(image_redimensionnee.shape)
        demi_image = image_redimensionnee[:250,:,:]

        image_miroir = np.concatenate((np.flip(demi_image,axis=0),demi_image),axis=0)
        # print('len truc',np.flip(demi_image,axis=0).shape   ,demi_image.shape, image_miroir.shape)
        # cv2.imshow('frame',cv2.resize(cv2.rotate(image_couleur/255.0,cv2.ROTATE_90_COUNTERCLOCKWISE),(800,500)))
        # print(image_redimensionnee.shape,cv2.cvtColor(cercles,cv2.COLOR_GRAY2BGR).shape)

        image_melangee = cv2.multiply(image_miroir,cercles,dtype=cv2.CV_64F)
        # print(image_redimensionnee.dtype,cercles.dtype,image_melangee.dtype)
        cv2.imshow('frame',image_melangee)
        
        fin = time.time()
        cv2.waitKey(5)
        # print('image_couleur.shape',image_couleur.shape)
    else:
        # image_normee = np.clip(image_normee*255,0,255)
        # item_image.setImage(image_normee,autoLevels=False)
        # print('Etape X: afficher l\'image en noir et blanc')
        fin = time.time()
        # cv2.imshow('frame',carre_blanc)
        # cv2.waitKey(20)
        # print('image_normee.shape',image_normee.shape)
    


    print('tracer',np.int(np.round((fin-debut)*1000)),'ms')





# la
def mettre_a_jour():
    global memoire, spectre, phase, temps_derniere_mesure, carre_blanc, enveloppe
    # print('Etape 1: lire le son')
    data = stream.read(echantillons_par_tampon,exception_on_overflow = False)
    
    temps_derniere_mesure = time.time()
    # data = struct.unpack(str(2 * echantillons_par_tampon) + 'B', data)
    # data = np.array(data, dtype='b')[::2] + 128
    

    
    sons = (np.frombuffer(data, dtype=np.int16) ) / 1024 / 32
    son = sons[0::2]
    son2 = sons[1::2]
    # print('Len(son)',len(son))
    maximum = np.mean(np.abs(son2))
    enveloppe = np.roll(enveloppe,shift=1)
    nouveau_cercle = np.minimum(maximum * 255 / 20,1)*3+1
    enveloppe[0] = nouveau_cercle
    print('Valeur maximum :',maximum) # valeur maximum : 32
    luminosite = int(maximum * 255)
    carre_blanc = np.ones(shape=(100,100)) * luminosite / 100
    # cv2.imshow('carre',carre_blanc)
    # cv2.waitKey(5)
    # fenetre.setBackground((luminosite,luminosite,luminosite))
    # data = echantillons_par_tampon * data
    memoire = np.concatenate((memoire,son))[-(echantillonage*10):]
    # on ne garde que les dernieres 10 secondes en mémoire

    nouveau_spectre, frequences_pixels, nouvelle_phase = spectre_complet(memoire)

    
    
    if len(phase) != len(nouvelle_phase):
        phase = np.zeros(len(nouvelle_phase))
    if len(spectre) != len(nouveau_spectre):
        spectre = np.zeros(len(nouveau_spectre))
    
    phase = (1 - facteur_lissage) * nouvelle_phase + facteur_lissage * phase
    spectre = (1 - facteur_lissage) * nouveau_spectre + facteur_lissage * spectre


    tracer(schema='image', data_x=frequences_pixels, data_y=spectre)
    os.system('clear')




#%%



avant = time.time()

while True:
    mettre_a_jour()
    # print('Temps depuis la derniere image',np.round((time.time()-avant)*1000,0),'ms')
    print('Fréquence de rafraichissement',np.round(1/(time.time()-avant),0),'Hz')
    avant = time.time()