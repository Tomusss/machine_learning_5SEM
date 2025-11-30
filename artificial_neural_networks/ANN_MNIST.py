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
    ### Uzupełnić

def softmax(z: np.ndarray, pochodna: bool=False) -> np.ndarray:
    '''
    z1,...,zn |-> (exp(z1)/sum(exp(zi)), ..., exp(zn)/sum(exp(zi)))
    '''
    ### Uzupełnić


def entropia_krzyzowa(y: np.ndarray, test: np.ndarray, pochodna: bool=False) -> float | np.ndarray:
    '''
    (y,test) |-> -sum(y*log(test))
    '''
    ###Uzupełnić

class warstwa():
    def __init__(self, n: int, f_aktywacji: Callable|None):
        self.n = n
        self.f_aktywacji = f_aktywacji
    def inicjuj(self, rozmiar_poprzedniej: int):
        m = rozmiar_poprzedniej
        self.wagi = np.random.normal(scale=np.sqrt(6/m), size=(m+1, self.n))

class warstwa_wejsciowa(warstwa):
    def __init__(self, n: int):
        super().__init__(n, None)
    def inicjuj(self, rozmiar_poprzedniej: int=0):
        pass
    def w_przod(self, wejscie: np.ndarray):
        if wejscie.shape[1]!=self.n:
            raise ValueError("Wielkość wektora wejściowego jest inna niż rozmiar warstwy")
        self.a=wejscie

class warstwa_wyjsciowa(warstwa):
    def __init__(self, n: int, f_aktywacji: Callable):
        super().__init__(n, f_aktywacji)
    def w_przod(self, a_poprzedniej):
        #Uzupełnić o obliczenia odpowiedzi warstwy na podstawie wektora zwracanego przez poprzednią warstwę
    def w_tyl(self, test: np.ndarray, a_poprzedniej: np.ndarray, krok: float):
        self.pochodna_z=1/len(test)*(self.a-test)
        #Uzupełnić o obliczenie gradientu funkcji straty względem wag dla wyznaczonej odpowiedzi a oraz oczekiwanej odpowiedzi test
        #Należy przechować wartości pochodnych względem odpowiedzi poprzedniej warstwy i zaktualizować wagi zgodnie z sgd 

class warstwa_ukryta(warstwa):
    def __init__(self, n: int, f_aktywacji: Callable = sigmoid):
        super().__init__(n, f_aktywacji)
    def w_przod(self, a_poprzedniej):
        #Uzupełnić o obliczenia odpowiedzi warstwy na podstawie wektora zwracanego przez poprzednią warstwę
    def w_tyl(self, pochodna_nastepnej: np.ndarray, a_poprzedniej: np.ndarray, krok: float):
        #Uzupełnić o obliczenie gradientu funkcji straty względem wag tej warstwy wykorzystując pochodną funkcji złożonej.
        #Należy przechować wartości pochodnych względem odpowiedzi poprzedniej warstwy i zaktualizować wagi zgodnie z sgd

class siec_neuronowa():
    def __init__(self, warstwy: tuple|list):
        m=warstwy[0].n
        self.warstwy=[]
        for warstwa in warstwy:
            warstwa.inicjuj(m)
            self.warstwy.append(warstwa)
            m=warstwa.n
    def predict(self, x):
        #Usupełnić o wyznaczanie odpowiedzi sieci dla zadanego wektora wejściowego
    def fit(self, X_train: np.ndarray, Y_train: np.ndarray,
            validation: bool = False, 
            X_val : np.ndarray | None = None, Y_val : np.ndarray | None = None,
            krok: float = 1, epochs : int = 20, batch_size: int = 200, plot : bool = False,
            dynamic_step: bool = True):
        #Uzupełnić o trenowanie sieci (wykorzystać metody w_przod() i w_tyl() warstw)
        if not validation:
            X_val, Y_val =X_train, Y_train
        else:
            if X_val is None or Y_val is None:
                raise ValueError("If validation is False, X_val and Y_val are required")
        N=X_train.shape[1]
        testy = []
        for k in range(1,epochs+1):
            y = self.predict(X_val)
            indeksy = np.random.choice(N,size=batch_size)
            print(f"Epoch: {k}")
            for i in tqdm(indeksy, desc='Postep'):
                ## Uzupełnić
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
    ann = siec_neuronowa([warstwa_wejsciowa(n=784), warstwa_ukryta(n=500),
                          warstwa_ukryta(n=500), warstwa_ukryta(n=500),
                          warstwa_wyjsciowa(n=10, f_aktywacji=softmax)])
    (X_train, y_train), (X_test, y_test) = mnist_load()
    ann.fit(X_train, y_train, validation = False, krok=0.1)
    Y_pred=np.argmax(ann.predict(X_test),axis=1)
    y_real=np.argmax(y_test, axis=1)
    #Y_pred=ann.predict(X_test)
    for predicted, real in zip(Y_pred,y_real):
        print(f'Przewidziane: {predicted}, Prawdziwe: {real}')

if __name__=="__main__":
    main()
