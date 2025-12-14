import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
from keras.datasets import mnist
from keras.utils import to_categorical
from tqdm import tqdm

def sigmoid(z: np.ndarray, pochodna: bool=False) -> np.ndarray:
    '''
    Jeśli pochodna jest False:
    z |-> 1/(1+exp(-z))
    Jeśli pochodna jest True:
    z |-> (1/(1+exp(-z)))'
    '''
    z = np.clip(z, -500, 500) # obcianamy bo i tak sigm(-500 to zero, a 500 to jeden)
    z = 1/(1+np.exp(-z))
    if pochodna:
        return(z*(1-z))
    
    return(z)
    
def relu(z, pochodna=False, alpha=0.01):
    if not pochodna:
        return np.maximum(z, alpha * z)
    
    dz = np.copy(z) 
    
    # Warunek 1: Dla z > 0, pochodna wynosi 1
    dz[z > 0] = 1
    
    # Warunek 2: Dla z <= 0, pochodna wynosi alpha (0.01)
    # W NumPy stosujemy <= aby objąć również punkt 0
    dz[z <= 0] = alpha
    
    return dz

def softmax(z: np.ndarray, pochodna: bool=False) -> np.ndarray:
    '''
    z1,...,zn |-> (exp(z1)/sum(exp(zi)), ..., exp(zn)/sum(exp(zi)))
    '''
    #z = z - np.max(z, axis=1, keepdims=True)
    z_exp = np.exp(z)
    s = z_exp / np.sum(z_exp, axis=1, keepdims=True)
    return s


def entropia_krzyzowa(y: np.ndarray, test: np.ndarray, pochodna: bool=False) -> float | np.ndarray:
    '''
    (y,test) |-> -sum(test*log(y))
    '''
    # zabezpiecznie przed log(0)
    eps = 1e-12
    y = np.clip(y, eps, 1 - eps)
    z= -np.sum(test*np.log(y)) / y.shape[0]

    if pochodna:
        # nie uzywamy do propagacji wstecznej
        # liczymy recznie pozniej 
        return(z)
    return(z)

class warstwa():
    def __init__(self, n: int, f_aktywacji: Callable|None):
        self.n = n
        self.f_aktywacji = f_aktywacji
    def inicjuj(self, rozmiar_poprzedniej: int):
        # inicjucjemy tutaj, tak aby wszystkie warstwy były już utworzone
        m = rozmiar_poprzedniej # na wejśiciu 
        if self.f_aktywacji == relu:
            self.wagi = np.random.normal(0, np.sqrt(2/m), size=(m+1, self.n))
        else:
            self.wagi = np.random.normal(scale=np.sqrt(6/m), size=(m+1,self.n))

class warstwa_wejsciowa(warstwa):
    def __init__(self, n: int):
        super().__init__(n, None) # brak funkcji aktywacji, przetwarzamy jedynie dane 
    def inicjuj(self, rozmiar_poprzedniej: int=0):
        # brak poprzedniej
        pass
    def w_przod(self, wejscie: np.ndarray):
        if wejscie.shape[1]!=self.n:
            raise ValueError("Wielkość wektora wejściowego jest inna niż rozmiar warstwy")
        self.a=wejscie

class warstwa_wyjsciowa(warstwa):
    def __init__(self, n: int, f_aktywacji: Callable):
        super().__init__(n, f_aktywacji)

    def w_przod(self, a_poprzedniej):
        ones = np.ones((a_poprzedniej.shape[0],1))  # tworzymy (n,1)
        a_bias = np.hstack((a_poprzedniej,ones))    # mamy macierz (m+1,n)
        self.z = a_bias @ self.wagi
        self.a = self.f_aktywacji(self.z)

        return(self.a)

    def w_tyl(self, test: np.ndarray, a_poprzedniej: np.ndarray, krok: float):
        self.pochodna_z=1/len(test)*(self.a - test)
        ones = np.ones((a_poprzedniej.shape[0],1))  # tworzymy (n,1)
        a_bias = np.hstack((a_poprzedniej,ones))    # mamy macierz (m+1,n)

        grad = a_bias.T @ self.pochodna_z   # dL/dW = dL/dZ * dZ/dW (dL/dZ oblicznone, dZ/dW to a.T)
        pochodna_w_tyl = self.pochodna_z @ self.wagi[:-1,:].T    # musimy zwrocic do tyłu błąd * waga dla kazdego neuronu

        self.wagi -= krok * grad

        return(pochodna_w_tyl)

class warstwa_ukryta(warstwa):
    def __init__(self, n: int, f_aktywacji: Callable = relu):
        super().__init__(n, f_aktywacji)

    def w_przod(self, a_poprzedniej):
        ones = np.ones((a_poprzedniej.shape[0],1))
        a_bias = np.hstack((a_poprzedniej,ones))

        self.z = a_bias @ self.wagi
        self.a = self.f_aktywacji(self.z)

        return(self.a)
    
    def w_tyl(self, pochodna_nastepnej: np.ndarray, a_poprzedniej: np.ndarray, krok: float):
        ones = np.ones((a_poprzedniej.shape[0],1))
        a_bias = np.hstack((a_poprzedniej,ones))
        da = self.f_aktywacji(self.z, pochodna = True) 
        self.pochodna_z = pochodna_nastepnej * da   # dL/dZ = dL/dA * dA/dZ , mnozenie zwykle bo f.aktywacji jest po kazdym elemencie

        grad = a_bias.T @ self.pochodna_z
        pochodna_w_tyl = self.pochodna_z @ self.wagi[:-1,:].T
        self.wagi -= krok * grad

        return(pochodna_w_tyl)
class siec_neuronowa():
    def __init__(self, warstwy: tuple|list):
        m=warstwy[0].n
        self.warstwy=[]
        for warstwa in warstwy:
            warstwa.inicjuj(m)
            self.warstwy.append(warstwa)
            m=warstwa.n

    def predict(self, x):
        self.warstwy[0].w_przod(x)
        sygnal = self.warstwy[0].a 
        for i in range(1,len(self.warstwy)):
            self.warstwy[i].w_przod(sygnal)
            sygnal = self.warstwy[i].a
        return(sygnal)

    def fit(self, X_train: np.ndarray, Y_train: np.ndarray,
            validation: bool = False, 
            X_val : np.ndarray | None = None, Y_val : np.ndarray | None = None,
            krok: float = 0.1, epochs : int = 20, batch_size: int = 100, plot : bool = False,
            dynamic_step: bool = True):
        if not validation:
            X_val, Y_val =X_train, Y_train
        else:
            if X_val is None or Y_val is None:
                raise ValueError("If validation is False, X_val and Y_val are required")
        N=X_train.shape[0]
        testy = []
        for k in range(1,epochs+1):
            if dynamic_step:
                step = krok * (0.95)**(k-1) 
            U = N//batch_size
            indeksy = np.random.permutation(N)
            print(f"Epoch: {k}")
            for i in tqdm(range(U), desc='Postep'):
                batch=indeksy[i*batch_size:(i+1)*batch_size]
                xb = X_train[batch,:]
                yb = Y_train[batch,:]

                self.predict(xb)    #do forward pass wykorzystujemy gotowa metode, ktora od razu przejdzie i zaktualizuje self.a
                ost_grad = self.warstwy[-1].w_tyl(yb, self.warstwy[-2].a,step)

                for war in range(len(self.warstwy)-2, 0,-1): # teraz backward pass 
                    ost_grad = self.warstwy[war].w_tyl(ost_grad,self.warstwy[war-1].a, step)
            y = self.predict(X_val)
            loss_function_val=entropia_krzyzowa(y,Y_val)
            print(f"Epoch: {k}, Loss function for validation vector: {loss_function_val:.4f}")
            if plot:
                testy.append(loss_function_val)
        if plot:
            plt.plot(np.arange(epochs),testy,ls='-.')
            plt.title('Value of the loss function for validation vector after each epoch')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.show()
    
def mnist_load():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train_r = X_train.reshape(60000, 784)
    X_test_r = X_test.reshape(10000, 784)
    X_train_r = X_train_r.astype('float32')
    X_test_r = X_test_r.astype('float32')
    X_train_r /= 255                   
    X_test_r /= 255
    liczba_klas = 10
    y_train = to_categorical(y_train, liczba_klas)
    y_test = to_categorical(y_test, liczba_klas)
    return (X_train_r, y_train), (X_test_r, y_test)

def main():
    #ann = siec_neuronowa([warstwa_wejsciowa(n=784), warstwa_ukryta(n=512),warstwa_ukryta(n=256), warstwa_wyjsciowa(n=10, f_aktywacji=softmax)])
    ann = siec_neuronowa([warstwa_wejsciowa(n=784), warstwa_ukryta(n=500),warstwa_ukryta(n=200), warstwa_ukryta(n=200), warstwa_wyjsciowa(n=10, f_aktywacji=softmax)])

    (X_train, y_train), (X_test, y_test) = mnist_load()
    ann.fit(X_train, y_train, validation = False, krok=0.05)
    Y_pred=np.argmax(ann.predict(X_test),axis=1)
    y_real=np.argmax(y_test, axis=1)
    #Y_pred=ann.predict(X_test)
    counter = 0
    for predicted, real in zip(Y_pred,y_real):
        if predicted == real: 
            counter+=1
        else:
            print("-----")
        print(f'Przewidziane: {predicted}, Prawdziwe: {real}')
    print(counter/len(Y_pred))
if __name__=="__main__":
    main()
