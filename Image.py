from enum import Enum

prohibicion = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]
peligro = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
stop = [14]
direccion_prohibida = [17]
ceda_paso = [13]
direccion_obligatoria = [38]


class SignalType(Enum):
    PROHIBIDO = 1
    PELIGRO = 2
    STOP = 3
    DIRECCION_PROHIBIDA = 4
    CEDA_EL_PASO = 5
    DIRECCION_OBLIGATORIA = 6
    NO_SEÑAL = 0


def return_type(type: int):
    '''
    Devuelve el enum para un tipo de señal concreto
    Parameters
    ----------
    type : int
        numero clasificador de la señal
    Returns
    -------
    Enum
        Objeto enum que identifica a la señal
    '''
    if type in prohibicion:
        return SignalType.PROHIBIDO
    elif type in peligro:
        return SignalType.PELIGRO
    elif type in stop:
        return SignalType.STOP
    elif type in direccion_prohibida:
        return SignalType.DIRECCION_PROHIBIDA
    elif type in ceda_paso:
        return SignalType.CEDA_EL_PASO
    elif type in direccion_obligatoria:
        return SignalType.DIRECCION_OBLIGATORIA
