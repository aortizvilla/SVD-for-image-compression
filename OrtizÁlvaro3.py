import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.linalg import lu, qr, svd, norm

A = plt.imread('OrtizAlvaro.jpeg')
plt.imshow(A)

# Considerem les següents funcions plantejades en l'exercici anterior:
    
def cost_short(A):
    m, n = A.shape
    U, S, Vt = svd(A)
    print("La imatge té %d valors singulars" % S.size)
    cost_original = 64*m*n
    print('Cost emmagatzemar la imatge original: %d' % cost_original)    
    k= S.size 
    cost_compressio = k *(m+n)*64
    print('Cost emmagatzemar la imatge comprimida utilitzant els valors singulars més grans: %d' % cost_compressio)
    percentatge = (cost_compressio/cost_original)*100
    print('Percentatge de compresió màxim: %d' %percentatge)
    return percentatge

def eSVD_rank(S, k):
    eSVD= S[k]/S[0]
    return eSVD

def compress(Abw, percentatge):
    U,S, Vt = svd(Abw)
    m,n=Abw.shape
    rang_original= S.size
    print("La imatge té %d valors singulars" % S.size)
    cost_original = 64*m*n
    
    rang_reduit = math.ceil((percentatge*m*n)/(100*(m+n)))
    print ("El rang de la compressió és:", rang_reduit)
    k=rang_reduit
    ASVD = U[:, 0:k]@np.diag(S[0:k])@Vt[0:k, :]
    cost_compressio = k *(m+n)*64
    
    print('Cost emmagatzemar la imatge original: %d' % cost_original)
    print('Cost emmagatzemar la imatge comprimida %d' % cost_compressio)
    eSVD=eSVD_rank(S, k)
    
    return ASVD, rang_original, rang_reduit


# 12- 13.

cost_short(A[:, :, 0])
 
A_mida=A.shape
if len(A_mida)==3:
    m, n, rgb = A_mida
    print("La imatge carregada utilitza una escala RGB")
else:
    raise Exception("La imatge carregada no utilitza una escala RGB")

M=[]
error = np.zeros(3)
per = np.zeros(3)
for i in range(3):
    input_compressio=input("Introdueix el percentatge de compressió de la imatge:")
    percentatge=float(input_compressio)
    per[i]=percentatge
    ASVD = np.zeros(A_mida, dtype=np.uint8)
    eSVD=0.0 #Inicialitzem l'error global de la compressió

    #Comprimim la imatge per a cada capa:
    for color in range(0,rgb):
        #Prenem la matriu corresponent al color:
        Acolor = A[:, :, color]
        #Apliquem la compressió de la matriu al percentatge introduït:
        Acolor_comprimida, r0, r1 = compress(Acolor, percentatge)
        ASVD[:, :, color]= Acolor_comprimida 
        eSVD_color=norm(Acolor-Acolor_comprimida, 2)/norm(Acolor, 2)
        eSVD = eSVD + eSVD_color**2
        print('Error relatiu (eSVD) de la capa %d: %5.4f' %(color+1, eSVD_color))
    M.append(ASVD)
    #L'error relatiu global és:
    eSVD=math.sqrt(eSVD)
    error[i]=eSVD
    print('Errror relatiu global de la compressió: %5.4f' %eSVD)
    

fig = plt.figure()

fig.add_subplot(2,2,1)
plt.imshow(A)
plt.axis('off')
plt.title('Imatge original')

fig.add_subplot(2,2,2)
plt.imshow(M[0])
plt.axis('off')
plt.title('SVD amb compressió al %d\n per cent i eSVD %5.4f' %(per[0], error[0]))

fig.add_subplot(2,2,3)
plt.imshow(M[1])
plt.axis('off')
plt.title('SVD amb compressió al %d\n  per cent i eSVD %5.4f' %(per[1], error[1]))

fig.add_subplot(2,2,4)
plt.imshow(M[2])
plt.axis('off')
plt.title('SVD amb compressió al %d\n  per cent i eSVD %5.4f' %(per[2], error[2]))

plt.tight_layout(pad=1.7)
plt.show()