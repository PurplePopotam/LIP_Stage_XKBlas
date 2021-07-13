# Produit matriciel

## Compiler

D'abord faire un 'module load intel-mkl/2018.1.163_gcc-6.4.0'. Puis 'make clean' et 'make'. Cela va générer un main. 

## Run

Il faut passer la taille des matrices en argument. Les matrices sont carrées et de taille identique. Par exemple pour multiplier deux matrices de taille 1000 : './main 1000'.

## Sortie 

Le programme va afficher un tableau à 5 colonnes :

* De quelle itération il s'agit
* Le temps d'initialisation (remplissage) des matrices sur le CPU.
* Le temps d'exécution de la version tiled sur GPU.
* Le temps d'exécution de la version 4 sur GPU (qui ne fait pas intervenir de tile).
* Le temps de rapatriement des données depuis le GPU sur l'hôte.

On n'affiche pas/ne fait pas le calcul sur CPU parce que cela prend rapidement beaucoup de temps. 

## Paramètres

* On peut vérifier que les deux résultats sont égaux : variable 'debug' ligne 36 du main.cu.
* On peut changer le nombre d'itérations : variable 'ITER' ligne 37 du main.cu.
* On peut changer la taille des blocks : variable 'THREADS_NUMBER' ligne 7 du kernels.cuh.

## Noyaux

Il y a plusieurs versions du produit matriciel, pour plus de détails voir kernels.cuh/kernels.cu.
Il y a aussi une addition de matrice. L'appel aux noyaux se fait dans le main.cu : ligne 84 et 96.
