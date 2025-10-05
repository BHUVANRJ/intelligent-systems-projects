import math
import numpy as np

class Entropy:
    def __init__(self, C, S):
        """
        C = number of classes
        S = total number of samples
        values will be asked from user
        """
        self.C = C
        self.S = S

        # ask for class label counts
        print(f"Enter {self.C} class label counts (one per line):")
        self.values = np.array([int(input()) for _ in range(self.C)])

        # probabilities
        self.Psamples = self.values / self.S

        # entropy calculation
        self.entropy = 0
        for pi in self.Psamples:
            if pi > 0:   # avoid log(0)
                self.entropy += -pi * np.log2(pi)

    def show(self):
        print("Class counts:", self.values)
        print("Probabilities:", self.Psamples)
        print("Entropy:", self.entropy)
        
        
class Gain:
    def __init__(self, entropyE, S, P, valuesSV):
        self.entropyE = entropyE
        self.valuesSV = np.array(valuesSV)
        self.S=S
        self.P=P
        self.gain=0
        for i in range(P):
            entropyi=Entropy(C, valuesSV[i])
            self.gain += ((valuesSV[i]/S) * entropyi.entropy)
        self.gain = entropyE.entropy - self.gain
    def show(self):
        print("Gain:", self.gain)            
            
        
    
    
    
    
    
    
    
C = int(input("Enter number of classes in target attribute: "))
S = int(input("Enter total number of data samples in Entire dataset: "))
entropyE = Entropy(C, S)
entropyE.show()

P = int(input("Enter number of classes label in choosen attribute(for gain) : "))
SV = int(input("Enter total number of data samples in DN:"))
print(f"Enter no of samples in the new choosen attribute's {P} class labels (one per line):")
valuesSV = [int(input()) for _ in range(P)]
gain1=Gain(entropyE, S, P, valuesSV)
gain1.show()