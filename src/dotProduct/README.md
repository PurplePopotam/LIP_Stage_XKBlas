# Produit Scalaire 

## Compiler 

Il y a juste à faire un 'make clean' puis 'make' dans le répertoire DotProduct (normalement c'est celui où vou sêtes actuellement). Cela va générer un main.

## Run

Il faut passer la taille des vecteurs en argument. 
Par exemple pour un produit scalaire de vecteurs de taille 1000  : './main 1000'

## Sortie 

Le programme affiche alors un tableau à 5 colonnes :

* De quelle itération il s'agit
* Le temps d'exécution sur CPU
* Le temps d'exécution sur GPU
* Le temps d'exécution sur GPU mesuré par l'hôte (prend en compte l'appel au noyau)
* Le temps de rapatriement des données depuis le GPU jusqu'à l'hôte.

## Paramètres

* On peut changer le nombre d'itération dans le main : variable 'ITER' ligne 9 du main.cu.
* On peut afficher le résultat des calculs : variable 'debug' ligne 31 du main.cu.
* On peut changer la taille des blocks : variable 'THREADS_PER_BLOCK' ligne 5 du kernels.cuh.

## Noyaux 

Il y a un produit terme à terme et 3 produit scalaire, pour plus de détails voir kernels.cuh/kernels.cu.
La version 4 ne marche pas.
L'appel à un noyau se fait ligne 89 du main.cu.
