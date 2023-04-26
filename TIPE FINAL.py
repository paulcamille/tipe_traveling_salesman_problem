import numpy as np
import random as rd
import matplotlib.pyplot as plt
## FONCTIONS UTILES

distance = lambda A,B : np.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)

def generer_pts(n,coord_max):                                                           # Pour generer aleatoire un ensemble de point par lesquels le drone doit passer
    pts = []
    while len(pts) < n:
        x = rd.randrange(0,coord_max)
        y = rd.randrange(0,coord_max)
        if not ([x,y] in pts):
            pts += [[x,y]]
    return pts

def generer_ch_aléatoire(n):
    presents = []
    manquants = [True]*n
    while len(presents) < n:
        tmp = rd.randint(0,n-1)
        if manquants[tmp]:
            manquants[tmp] = False
            presents.append(tmp)
    return presents

def tabl_dist(pts):                                                                     # Matrice symetrique contenant les distance entre les différents points
    n = len(pts)
    tableau = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            dist = distance(pts[i],pts[j])
            tableau[i,j] = dist
            tableau[j,i] = dist
    return tableau

def longueur_ch(ch,tableau):                                                            # Calculer la longueur d'un chemin
    long = 0
    p0 = len(tableau)-1
    for p1 in ch:
        long += tableau[p0,p1]
        p0 = p1
    return long

## ALGORITHME GLOUTON

def algo_glouton(pts):                                                                  # Generer un chemin par la éthode du plus proche voisin
    n =len(pts)
    tableau = tabl_dist(pts)
    long_opt = np.inf
    ch_opt = []
    for k in range(n):
        visités = [k]
        manquants = [True]*n
        manquants[k] = False
        for l in range(n-1):
            dist_proche = np.inf
            for t in range(n):
                d = tableau[visités[l],t]
                if d <= dist_proche and manquants[t]:
                    dist_proche = d
                    indice = t
            visités.append(indice)
            manquants[indice] = False
        long = longueur_ch(visités,tableau)
        if long <= long_opt:
            long_opt = long
            ch_opt = visités
    return [long_opt,ch_opt]

##

def fusion(l1,l2):
    if len(l1)==0:
        return l2
    if len(l2)==0:
        return l1
    if l1[0][0]>=l2[0][0]:
        return [l2[0]]+fusion(l1,l2[1::])
    else:
        return [l1[0]]+fusion(l1[1::],l2)

def separe(l):
    if len(l)<2:
        return l
    else:
        u=l[:(len(l)//2)]
        v=l[len(l)//2:]
        return [u,v]

def tri_fusion(l):
    if len(l)<2:
        return l
    else:
        [u,v]=separe(l)
        return fusion(tri_fusion(u),tri_fusion(v))

## ALGORITHME GENETIQUE

def creer_population(m,tableau):
    population = []
    ch = list(range(len(tableau)))
    for i in range(m):
        rd.shuffle(ch)
        longueur = longueur_ch(ch,tableau)
        population.append([longueur,ch])
    return population

def reduire(population):
    l = tri_fusion(population)
    population[:] = l[:len(population)//2]

def normaliser_ch(ch,n):
    presents = []
    manquants = [True]*n
    for p in ch:
        if p < n and manquants [p]:
            presents.append(p)
            manquants[p] = False
    for i in range(n):
        if manquants[i]:
            presents.append(i)
    return presents

def muter_ch(ch):
    n = len(ch)
    i = j = rd.randrange(0,n)
    while j == i:
        j = rd.randrange(0,n)
    ch[i],ch[j] = ch[j],ch[i]

def muter_population(population,proba,tableau):
    for i in range(1,len(population)):
        if rd.random() < proba:
            ch = population[i][1]
            muter_ch(ch)
            population [i] = [longueur_ch(ch,tableau),ch]

def croiser(c1,c2):
    n = len(c1)
    return normaliser_ch(c1[:n//2]+c2[n//2:],n)

def nouvelle_generation(population,tableau):
    n = len(population)
    for i in range(n-1):
        ch = croiser(population[i][1],population[i+1][1])
        population.append([longueur_ch(ch,tableau),ch])
    ch = croiser(population[0][1],population[n-1][1])
    population.append([longueur_ch(ch,tableau),ch])

def algo_genetique(pts,nb_pop,proba,gen):
    tableau = tabl_dist(pts)
    population = creer_population(nb_pop,tableau)
    for i in range(gen):
        reduire(population)                                                                 # Selection
        nouvelle_generation(population,tableau)                                             # Croisement
        muter_population(population,proba,tableau)                                          # Mutation
    return tri_fusion(population)[0]

## EXPERIENCES

def trace_compare(pts,ch1,ch2):
    tabl = tabl_dist(pts)
    n = len(ch1)
    x1=[]
    y1=[]
    x2=[]
    y2=[]
    for j in range(n):
        k1=ch1[j]
        x1+=[pts[k1][0]]
        y1+=[pts[k1][1]]
        k2=ch2[j]
        x2+=[pts[k2][0]]
        y2+=[pts[k2][1]]

    plt.figure()
    plt.subplot(1,2,1)
    plt.scatter(x1,y1,color='red')
    plt.plot(x1,y1,label=longueur_ch(ch1,tabl))
    plt.title('parcours sur chemin aléatoire 1')   # à completer le titre
    plt.legend()
    plt.grid()


    plt.subplot(1,2,2)
    plt.scatter(x2,y2,color='red')
    plt.plot(x2,y2,label=longueur_ch(ch2,tabl))
    plt.title('parcours sur chemin aléatoire 2')   # à completer le titre
    plt.legend()
    plt.grid()

    plt.show()

def trace_chemin(pts,ch):
    tabl = tabl_dist(pts)
    n = len(ch)
    x=[]
    y=[]
    for j in range(n):
        k=ch[j]
        x+=[pts[k][0]]
        y+=[pts[k][1]]
    plt.scatter(x,y,color='red')
    plt.plot(x,y,label=longueur_ch(ch,tabl))
    plt.title("chemin construit par l'algorithme glouton")                                  # à completer le titre
    plt.legend()
    plt.grid()
    plt.show()

def experience(liste,m):

    Y_plus = []
    Y_moy = []
    Y_moins = []
    moy = 0
    for k in range(len(liste)):
        print(k)
        moy = 0
        min = np.inf
        max = -np.inf
        for i in range(m):
            tmp = algo_glouton(generer_pts(liste[k],m))[0]
            moy += tmp
            if max<tmp:
                max = tmp
            if min>tmp:
                min =tmp
        Y_plus += [max]
        Y_moy += [moy/m]
        Y_moins += [min]

    plt.plot(liste,Y_plus,color='chartreuse')
    plt.plot(liste,Y_moy,label = 'dist moy glt',color = 'red')
    plt.plot(liste,Y_moins,color='chartreuse')
    plt.fill_between(liste, Y_plus, Y_moins, color='turquoise')
    plt.grid()
    plt.legend()
    plt.show()

def regres(l1,l2,n):
    coef = np.polyfit(l1,l2,n)
    f = np.poly1d(coef)
    return coef,f

def experience_reg(liste,m):

    Y_plus = []
    Y_moy = []
    Y_moins = []
    moy = 0
    for k in range(len(liste)):
        print(liste[k])
        moy = 0
        min = np.inf
        max = -np.inf
        for i in range(m):
            tmp = algo_glouton(generer_pts(liste[k],m))[0]
            moy += tmp
            if max<tmp:
                max = tmp
            if min>tmp:
                min =tmp
        Y_plus += [max]
        Y_moy += [moy/m]
        Y_moins += [min]

    pts = liste
    moy_pts = Y_moy
    reg = regres(pts,moy_pts,1)
    func = reg[1]

    x = np.linspace(pts[0],pts[len(pts)-1],5*len(pts))
    y = func(x)

    plt.plot(liste,Y_plus,color='chartreuse')
    plt.plot(liste,Y_moins,color='chartreuse')
    plt.fill_between(liste, Y_plus, Y_moins, color='turquoise')

    plt.plot(x,y,color='black',label='regression')
    plt.plot(pts,moy_pts,'x',color='red',label='points moyens')

    plt.grid()
    plt.legend()
    plt.show()

    return reg

def algo_genetique_modifie(pts,nb_pop,proba,gen):
    l = []
    tableau = tabl_dist(pts)
    population = creer_population(nb_pop,tableau)
    for i in range(gen):
        reduire(population)
        l+=[population[0][0]]
        nouvelle_generation(population,tableau)
        muter_population(population,proba,tableau)
    reduire(population)
    l+=[population[0][0]]
    return l

def trace_algo_gen_convergence(ens_pts,nb_pop,proba,gen):
    absc = list(range(1,gen+2))
    genetique = []
    for ens in ens_pts:
        ord = algo_genetique_modifie(ens,nb_pop,proba,gen)
        genetique.append(ord)
        print(len(absc)) #
        print(len(ord)) #
        plt.plot(absc,ord,label =(len(ens), 'points'))
    plt.legend()
    plt.title('Concergence des generations')
    plt.grid()
    plt.show()
##
n = 100
proba = 0.5
nb_population = 100
gen = 5000
pts = generer_pts(n,n)
# chemin_aléatoire1 = generer_ch_aléatoire(n)
# chemin_aléatoire2 = generer_ch_aléatoire(n)
glou = algo_glouton(pts)
gen = algo_genetique(pts,nb_population,proba,gen)

ens = [generer_pts(25,100),generer_pts(50,100),generer_pts(75,100),generer_pts(100,100)]
##
trace_chemin(pts,chemin_aléatoire)
##


chemin_aléatoire1 = [32, 28, 96, 12, 78, 60, 62, 98, 46, 67, 14, 55, 83, 50, 74, 71, 4, 53, 65, 80, 70, 35, 33, 79, 30, 40, 38, 76, 7, 84, 19, 43, 11, 5, 24, 66, 44, 1, 20, 17, 51, 82, 58, 47, 92, 25, 10, 16, 18, 68, 48, 97, 15, 93, 49, 45, 0, 52, 77, 89, 95, 31, 13, 2, 59, 42, 6, 29, 36, 57, 90, 64, 34, 94, 56, 69, 3, 41, 99, 21, 73, 85, 75, 37, 39, 61, 23, 91, 88, 22, 87, 54, 81, 72, 27, 86, 8, 63, 26, 9]

chemin_aléatoire2 = [92, 64, 5, 65, 54, 21, 31, 75, 63, 73, 82, 99, 51, 30, 1, 36, 94, 47, 7, 48, 45, 84, 19, 49, 98, 70, 85, 81, 50, 3, 71, 32, 25, 97, 57, 24, 93, 44, 58, 13, 80, 91, 69, 27, 8, 34, 88, 53, 90, 38, 20, 29, 23, 0, 33, 87, 37, 56, 68, 17, 78, 11, 83, 39, 89, 9, 2, 43, 18, 40, 4, 6, 55, 60, 61, 15, 12, 86, 59, 67, 76, 46, 52, 10, 72, 74, 41, 16, 95, 14, 26, 62, 77, 42, 96, 66, 35, 22, 79, 28]

pts = [[93, 14], [26, 89], [52, 99], [48, 90], [6, 11], [39, 89], [0, 40], [86, 85], [29, 68], [79, 3], [97, 18], [53, 19], [94, 32], [0, 1], [89, 5], [69, 41], [19, 73], [49, 38], [10, 80], [87, 31], [35, 3], [19, 28], [60, 14], [60, 42], [0, 13], [95, 5], [7, 9], [74, 0], [35, 30], [10, 40], [84, 72], [9, 79], [45, 78], [70, 95], [17, 80], [26, 24], [92, 65], [16, 69], [1, 16], [69, 73], [38, 77], [1, 51], [80, 35], [72, 92], [4, 86], [87, 65], [75, 76], [46, 13], [61, 7], [22, 8], [72, 5], [61, 2], [66, 83], [90, 13], [46, 75], [85, 7], [34, 88], [74, 29], [61, 16], [33, 98], [99, 6], [43, 36], [56, 99], [72, 97], [70, 16], [90, 48], [55, 0], [37, 45], [20, 34], [55, 52], [14, 66], [22, 45], [63, 51], [47, 45], [28, 41], [71, 60], [80, 40], [1, 22], [7, 98], [73, 1], [6, 46], [60, 24], [88, 4], [0, 8], [25, 96], [64, 28], [31, 74], [31, 71], [57, 22], [66, 33], [74, 70], [19, 63], [46, 77], [2, 80], [69, 50], [78, 16], [1, 20], [99, 13], [29, 96], [39, 2]]


##
glou = algo_glouton(pts)
##

liste = list(range(20,100))
m = 100
experience_reg(liste,m)

##
gen = 100000
proba = 0.5
nb_population = 800
# b = algo_genetique(pts,nb_population,proba,gen)

# enspts = [generer_pts(25,100),generer_pts(50,100),generer_pts(75,100),generer_pts(100,100)]
# trace_algo_gen_convergence(enspts,nb_population,proba,gen)

liste = list(range(25,50))
m = 100
glouton_vs_genetique(liste,m,pts,nb_population,proba,gen)

##

def glouton_vs_genetique(liste,m,pts,nb_population,proba,gen):

    Y_plus = []
    Y_moy = []
    Y_moins = []

    Y_genetique = []

    moy = 0
    for k in range(len(liste)):
        print(liste[k])
        moy = 0
        min = np.inf
        max = -np.inf
        for i in range(m):
            pts = generer_pts(liste[k],100)
            tmp = algo_glouton(pts)[0]
            moy += tmp
            if max<tmp:
                max = tmp
            if min>tmp:
                min =tmp
        Y_plus += [max]
        Y_moy += [moy/m]
        Y_moins += [min]

        Y_genetique += [algo_genetique(pts,nb_population,proba,gen)[0]]

    plt.plot(liste,Y_plus,color='chartreuse')
    plt.plot(liste,Y_moins,color='chartreuse')
    plt.fill_between(liste, Y_plus, Y_moins, color='turquoise')

    plt.plot(liste,Y_moy,label = 'distance_glouton',color = 'red')
    plt.plot(liste,Y_genetique,label = 'distance_genetique',color = 'black')



    plt.grid()
    plt.legend()
    plt.show()
##

def trace(pts,ch):
    n = len(ch)
    x=[]
    y=[]

    for j in range(n):
        k=ch[j]
        x+=[pts[k][0]]
        y+=[pts[k][1]]

    k_0=ch[0]
    x+=[pts[0][0]]
    y+=[pts[0][1]]

    plt.scatter(x,y,color='red')
    plt.plot(x,y,label=longueur_ch(ch,tabl)) # à completer le label
    plt.title('parcours aléatoire')
    plt.legend()
    plt.grid()

    plt.show()

def experience(liste,m):

    Y_plus = []
    Y_moy = []
    Y_moins = []
    moy = 0
    for k in range(len(liste)):
        moy = 0
        min = np.inf
        max = -np.inf
        for i in range(m):
            tmp = algo_glouton(generer_pts(liste[k],m))[0]
            moy += tmp
            if max<tmp:
                max = tmp
            if min>tmp:
                min =tmp
        Y_plus += [max]
        Y_moy += [moy/m]
        Y_moins += [min]

    plt.plot(liste,Y_plus,color='chartreuse')
    plt.plot(liste,Y_moy,label = 'dist moy glt',color = 'red')
    plt.plot(liste,Y_moins,color='chartreuse')
    plt.fill_between(liste, Y_plus, Y_moins, color='turquoise')
    plt.grid()
    plt.legend()
    plt.show()

def regres(l1,l2,n):
    coef = np.polyfit(l1,l2,n)
    f = np.poly1d(coef)
    return coef,f

def experience_reg(liste):

    Y_plus = []
    Y_moy = []
    Y_moins = []
    moy = 0
    for k in range(len(liste)):
        print(liste[k])
        moy = 0
        min = np.inf
        max = -np.inf
        for i in range(m):
            tmp = long_tot(glouton(genere(liste[k])))
            moy += tmp
            if max<tmp:
                max = tmp
            if min>tmp:
                min =tmp
        Y_plus += [max]
        Y_moy += [moy/m]
        Y_moins += [min]

    pts = liste
    moy_pts = Y_moy
    reg = regres(pts,moy_pts,1)
    func = reg[1]

    x = np.linspace(pts[0],pts[len(pts)-1],5*len(pts))
    y = func(x)

    plt.plot(liste,Y_plus,color='chartreuse')
    plt.plot(liste,Y_moins,color='chartreuse')
    plt.fill_between(liste, Y_plus, Y_moins, color='turquoise')

    plt.plot(x,y,color='black',label='regression')
    plt.plot(pts,moy_pts,'x',color='red',label='points moyens')

    plt.grid()
    plt.legend()
    plt.show()

    return reg

##
b = algo_genetique(pts,10,0.5,100000)

