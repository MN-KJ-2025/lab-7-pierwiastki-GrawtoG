# =================================  TESTY  ===================================
# Testy do tego pliku zostały podzielone na dwie kategorie:
#
#  1. `..._invalid_input`:
#     - Sprawdzające poprawną obsługę nieprawidłowych danych wejściowych.
#
#  2. `..._correct_solution`:
#     - Weryfikujące poprawność wyników dla prawidłowych danych wejściowych.
# =============================================================================
import numpy as np
import numpy.polynomial.polynomial as nppoly


def roots_20(coef: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Funkcja wyznaczająca miejsca zerowe wielomianu funkcją
    `nppoly.polyroots()`, najpierw lekko zaburzając wejściowe współczynniki 
    wielomianu (N(0,1) * 1e-10).

    Args:
        coef (np.ndarray): Wektor współczynników wielomianu (n,).

    Returns:
        (tuple[np.ndarray, np. ndarray]):
            - Zaburzony wektor współczynników (n,),
            - Wektor miejsc zerowych (m,).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    try:
        if not isinstance(coef,np.ndarray):
            return None
        
        if len(coef)<=0:
            return None
        for i,c in enumerate(coef):
            coef[i]=np.random.random_sample()*1e-10+c
        return coef, nppoly.polyroots(coef)
    except Exception:
        return None
    

def frob_a(coef: np.ndarray) -> np.ndarray | None:
    """Funkcja służąca do wyznaczenia macierzy Frobeniusa na podstawie
    współczynników jej wielomianu charakterystycznego:
    w(x) = a_n*x^n + a_{n-1}*x^{n-1} + ... + a_2*x^2 + a_1*x + a_0

    Testy wymagają poniższej definicji macierzy Frobeniusa (implementacja dla 
    innych postaci nie jest zabroniona):
    F = [[       0,        1,        0,   ...,            0],
         [       0,        0,        1,   ...,            0],
         [       0,        0,        0,   ...,            0],
         [     ...,      ...,      ...,   ...,          ...],
         [-a_0/a_n, -a_1/a_n, -a_2/a_n,   ..., -a_{n-1}/a_n]]

    Args:
        coef (np.narray): Wektor współczynników wielomianu (n,).

    Returns:
        (np.ndarray): Macierz Frobeniusa o rozmiarze (n,n).
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    try:
        if not isinstance(coef,np.ndarray):
            return None
        if len(coef)<=1:
            return None
        m=np.zeros([len(coef)-1,len(coef)-1])
        for i,w in enumerate(m):
            if i >= len(coef)-1:
                break
            w[-1]=-coef[(i)]/coef[-1]
            if i-1>=0:
                w[i-1]=1
            m[i]=w
        return np.transpose(m)
    except Exception:
        return None

print(frob_a(np.asarray([-0.0243451 ,  0.32057148, -1.4536215 ,  2.97211903, -2.81192549,1.])))


# [[ 0.        ,  1.        ,  0.        ,  0.        ,  0.        ],
#        [ 0.        ,  0.        ,  1.       ...   ,  0.        ,  0.        ,  1.        ],
#        [ 0.0243451 , -0.32057148,  1.4536215 , -2.97211903,  2.81192549]])
def is_nonsingular(A: np.ndarray) -> bool | None:
    """Funkcja sprawdzająca czy podana macierz NIE JEST singularna. Przy
    implementacji należy pamiętać o definicji zera maszynowego.

    Args:
        A (np.ndarray): Macierz (n,n) do przetestowania.

    Returns:
        (bool): `True`, jeżeli macierz A nie jest singularna, w przeciwnym 
            wypadku `False`.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """
    pass
