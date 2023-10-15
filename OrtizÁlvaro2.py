import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.linalg import lu, qr, svd, norm

def rgb2gray(A_rgb):
    return np.dot(A_rgb[...,:3], [0.2989, 0.5870, 0.1140])

A = plt.imread('OrtizAlvaro.jpeg')
plt.imshow(A)

Abw= rgb2gray(A)
plt.imshow(Abw, cmap='gray')


# 8) Expressió analítica per al càlcul de l'error relatiu eSVD de rang k.  
# Segons el teorema 3 la podem calcular de dues maneres. La primera funció
# descriu les dues: 
def eSVD_long(Abw):
    U,S, Vt = svd(Abw)
    rang_original = S.size 
    print("La imatge té %d valors singulars" % S.size)
    
    input_rang=input("Introdueix el rang de la matriu comprimida")
    rang_compressio =int(input_rang)
    if rang_compressio <=rang_original:
        print ("El rang de la compressió és:", rang_compressio)
    else:
        rang_compressio = rang_original
        print("El rang és superior al màxim, aleshores el rang rectificat és:", rang_compressio)

    k=rang_compressio
    ASVD = U[:, 0:k]@np.diag(S[0:k])@Vt[0:k, :]
    eSVD1=norm(Abw-ASVD, 2)/norm(Abw, 2)
    eSVD2= S[k]/S[0]
    print('SVD utilitzant %d valors singulars. Error relatiu (eSVD): %5.4f' %(k, eSVD1))
    return eSVD1, eSVD2

eSVD1, eSVD2=eSVD_long(Abw)
iguals=abs(eSVD1- eSVD2)<10e-14
print('Comprovació OK=', iguals)

def eSVD_rank(S, k):
    eSVD= S[k]/S[0]
    print('SVD utilitzant %d valors singulars. Error relatiu (eSVD): %5.4f' %(k, eSVD))
    return eSVD

# 9) Càlcul d'emmagatzematge d'una imatge: Original vs. SVD. La funció següent 
# calcula el cost d'emmagatzematge i el percentatge de compressió de l'aproximació
# SVD. 


# Cost màxim de compressió amb SVD:
def cost_short(Abw):
    U,S, Vt = svd(Abw)
    print("La imatge té %d valors singulars" % S.size)
    m,n = Abw.shape
    cost_original = 64*m*n
    print('Cost emmagatzemar la imatge original: %d' % cost_original)    
    k= S.size 
    cost_compressio = k *(m+n)*64
    print('Cost emmagatzemar la imatge comprimida utilitzant els valors singulars més grans: %d' % cost_compressio)
    percentatge = (cost_compressio/cost_original)*100
    print('Percentatge de compresió màxim: %d' %percentatge)
    return cost_original, cost_compressio, percentatge

# Cost adaptat a les k primeres columnes:
def cost_long(Abw):
    U,S, Vt = svd(Abw)
    print("La imatge té %d valors singulars" % S.size)
    
    m,n = Abw.shape
    cost_original = 64*m*n
    print('Cost emmagatzemar la imatge original: %d' % cost_original)
    
    rang_original = S.size 
    input_rang=input("Introdueix el rang de la matriu comprimida")
    rang_compressio =int(input_rang)
    if rang_compressio <=rang_original:
        print ("El rang de la compressió és:", rang_compressio)
    else:
        rang_compressio = rang_original
        print("El rang és superior al màxim, aleshores el rang rectificat és:", rang_compressio)
    
    k=rang_compressio
    cost_compressio = k *(m+n)*64
    print('Cost emmagatzemar la imatge comprimida utilitzant els valors singulars més grans: %d' % cost_compressio)
    
    percentatge = (cost_compressio/cost_original)*100
    print('Percentatge de compresió: %d' %percentatge)
    return cost_original, cost_compressio, percentatge

cost_long(Abw)

# 10) FUnció percentatge desitjat: Donat un percentatge calcula fins quin
# valor singular hem d'escollir. 
    
def compress(Abw, percentatge):
    U,S, Vt = svd(Abw)
    m,n=Abw.shape
    rang_original= S.size
    cost_original = 64*m*n
    
    rang_reduit = math.ceil((percentatge*m*n)/(100*(m+n)))
    print ("El rang de la compressió és:", rang_reduit)
    k=rang_reduit
    ASVD = U[:, 0:k]@np.diag(S[0:k])@Vt[0:k, :]
    cost_compressio = k *(m+n)*64
    
    print('Cost emmagatzemar la imatge original: %d' % cost_original)
    print('Cost emmagatzemar la imatge comprimida %d' % cost_compressio)
    eSVD=eSVD_rank(S, k)
    
    return ASVD, eSVD

ASVD1, eSVD1= compress(Abw, 25)
ASVD2, eSVD2= compress(Abw, 50)
ASVD3, eSVD3= compress(Abw, 75)
    

fig = plt.figure()

fig.add_subplot(2,2,1)
plt.imshow(Abw, cmap = 'gray')
plt.axis('off')
plt.title('Imatge original')

fig.add_subplot(2,2,2)
plt.imshow(ASVD1, cmap = 'gray')
plt.axis('off')
plt.title('SVD 25 per cent.\n Error relatiu= %5.4f' %eSVD1)

fig.add_subplot(2,2,3)
plt.imshow(ASVD2, cmap = 'gray')
plt.axis('off')
plt.title('SVD 50 per cent.\n Error relatiu=%5.4f' %eSVD2)

fig.add_subplot(2,2,4)
plt.imshow(ASVD3, cmap = 'gray')
plt.axis('off')
plt.title('SVD 75 per cent.\n Error relatiu=%5.4f' %eSVD3)

plt.tight_layout(pad=1.7)
plt.show()