from SlopeStability import CHI_PHI_SOIL,INFINITE_SLOPE,PURELY_COHESIVE_SOIL,PURELY_COHESIVE_SOIL_WITH_INCREASING_SHEAR_STRENGTH

import matplotlib.pyplot as plt
 
plt.style.use('Rplot')
 
# PURELY_COHESIVE_SOIL_WITH_INCREASING_SHEAR_STRENGTH(beta,H,H0,Cb,l,lb):
# PURELY_COHESIVE_SOIL_WITH_INCREASING_SHEAR_STRENGTH(45,100,15,1150,100,37.6)

# CHI_PHI_SOIL(beta,H,Hw,Hc,Hwdash,c,phi,l,lw,q,Ht):
# CHI_PHI_SOIL(30,23,0,0,0,280,17,120,62.5,0,0)

# INFINITE_SLOPE(beta,theeta,H,c,phi,cdash,phdash,l,lw,X,T):
# INFINITE_SLOPE(20,0,12,0,0,300,30,120,62.4,8,11.3)

# PURELY_COHESIVE_SOIL(beta,H,Hw,Hwdash,D,c,lw,l,Ht,q):
PURELY_COHESIVE_SOIL(beta=42,
                     H=13,
                     Hw=5,
                     Hwdash=0,
                     D=7,
                     c=410,
                     lw=62.5,
                     l=120,
                     Ht=0,
                     q=220)