import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lu, qr, svd, norm

def rgb2gray(A_rgb):
    return np.dot(A_rgb[...,:3], [0.2989, 0.5870, 0.1140])

A = plt.imread('OrtizAlvaro.jpeg')
plt.imshow(A)

Abw= rgb2gray(A)
plt.imshow(Abw, cmap='gray')

# 3) Realitzem les descomposicions QR, LU i SVD

P, L1, U1= lu(Abw)
Q, R= qr(Abw)
U,S, Vt = svd(Abw)

print("La imatge té %d valors singulars" % S.size)

#4) Prenem un valor k entre 1 i 1080 (min(1080, 1920))

rang_original = S.size 
input_rang=input("Introdueix el rang de la matriu comprimida")
rang_compressio =int(input_rang)
if rang_compressio <=rang_original:
    print ("El rang de la compressió és:", rang_compressio)
else:
    rang_compressio = rang_original
    print("El rang és superior al màxim, aleshores el rang rectificat és:", rang_compressio)

# Caculem ALU, AQR i ASVD 
k=rang_compressio
ALU =(P@ L1[:, 0:k])@U1[0:k,:]
AQR = Q[:,0:k]@R[0:k,:]
ASVD = U[:, 0:k]@np.diag(S[0:k])@Vt[0:k, :]

# 5) Calculem els errors relatius de les aproximacions en norma 2. 

# De teoria sabem que la norma 2 de la matriu és el primer valor singular. 
# Per tant, considerem norm(Abw, 2) = S[0]

eLU = norm(Abw-ALU, 2)/S[0]
eQR = norm(Abw-AQR, 2)/ S[0]
eSVD = norm(Abw-ASVD, 2)/S[0]

print('Error relatiu LU = %5.4f' % eLU)
print('Error relatiu QR = %5.4f' % eQR)
print('Error relatiu SVD = %5.4f' % eSVD)

# D'altra banda, pel teorema 3 sabem que norm(Abw-ASVD, 2)=S[k]. Ho podem fer
# quan k < S.size
if k<S.size:
    eSVD1= S[k]/ S[0]
    iguals=abs(eSVD- eSVD1)<10e-14
    print('Comprovació OK=', iguals)


# 6) Representem la imatge original i les tres imatges comprimides detallant els errors relaius 
# de les tres aproximacions. 


fig = plt.figure()

fig.add_subplot(2,2,1)
plt.imshow(Abw, cmap = 'gray')
plt.axis('off')
plt.title('Imatge original')

fig.add_subplot(2,2,2)
plt.imshow(ALU, cmap = 'gray')
plt.axis('off')
plt.title('LU utilitzant %d columnes.\n Error relatiu =%5.4f ' %(k, eLU))

fig.add_subplot(2,2,3)
plt.imshow(AQR, cmap = 'gray')
plt.axis('off')
plt.title('QR utilitzant %d columnes.\n Error relatiu =%5.4f' %(k, eQR))

fig.add_subplot(2,2,4)
plt.imshow(ASVD, cmap = 'gray')
plt.axis('off')
plt.title('SVD utilitzant %d columnes.\n Error relatiu =%5.4f' %(k, eSVD));

plt.tight_layout(pad=1.5)
plt.show()