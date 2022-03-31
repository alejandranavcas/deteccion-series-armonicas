#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 11:47:17 2022

@author: alejandranavarrocastillo
"""

# Main functions for "deteccion-series-armonicas" #

import numpy as np
import warnings


### get_spectrum ###

from numpy.fft import fft

def get_spectrum(y, srate, fft_len=None, win="hann"):
    """Obtain the windowed FFT of a signal"""
    if fft_len == None:
        fft_len = len(y)
        
    if win == "rect":
        wnd = np.ones(len(y))
    elif win == "hann":
        wnd = np.hanning(len(y))
    elif win == "hamming":
        wnd = np.hamming(len(y))
    elif win == "blackman":
        wnd = np.blackman(len(y))
    else:
        raise

    y2 = y*wnd
    out = fft(y2, fft_len)
    sp = abs(out[:int(fft_len/2)]/len(y))
    freq = np.linspace(0, srate/2, len(sp), endpoint=False)
    return freq, sp


### encontrar_picos ###

def encontrar_picos(amplitudes, altura_pico):
    """
    Encuentra los picos en una gráfica.
    
    Parámetros de entrada
    ---------
    amplitudes: lista
        Valores de la cual se quieren encontrar sus picos (máximos).
    altura_pico: float
        Altura minima que tiene que tener el pico para ser detectado.
    
    Valores de salida
    ---------
    peaks: lista
        Índices de la lista amplitudes en los cuales se encuentran los picos detectados.
    """
    peaks = []
    for i in range(1,len(amplitudes)-1):
        if amplitudes[i] >= altura_pico and amplitudes[i] > amplitudes[i-1] and amplitudes[i] >= amplitudes[i+1]:
                peaks.append(i)
    return peaks



### _peak_prominences ###

class PeakPropertyWarning(RuntimeWarning):
    """Calculated property of a peak has unexpected value."""
    pass

def _peak_prominences(amplitudes, peaks, wlen):
    """
    Calcula la prominencia de cada pico de la señal.
    Parametros
    ----------
    amplitudes : list
        Una lista de valores con picos.
    peaks : list
        Indices de los picos en `amplitudes`.
    wlen : np.intp
        Longitud de la ventana en número de muestras (ver `peak_prominences`) que se aproxima por 
        exceso al impar más cercano. Si wlen es más pequeño que 2 se usa la señal `amplitudes` entera.
        
    Valores de salida
    -------
    prominences : ndarray
        Las prominencias calculadas para cada pico en `peaks`.
    left_bases, right_bases : ndarray
        Las bases de cada pico como indices de `amplitudes` a la izquierda y derecha de cada pico.
    
    Raises
    ------
    ValueError
        Si el valor de `peaks` es un indice inválido para `amplitudes`.
    
    Warns
    -----
    PeakPropertyWarning
        Si la prominencia calculada de algún pico es 0.
    """
    
    show_warning = False
    prominences = np.empty(len(peaks), dtype=np.float64)
    left_bases = np.empty(len(peaks), dtype=np.intp)
    right_bases = np.empty(len(peaks), dtype=np.intp)
    
    for peak_nr in range(len(peaks)):
        peak = peaks[peak_nr]
        i_min = 0
        i_max = len(amplitudes) - 1
        if not i_min <= peak <= i_max:
            raise ValueError("peak {} is not a valid index for `amplitudes`".format(peak))
            
        if 2 <= wlen:
            # Adjust window around the evaluated peak (within bounds);
            # if wlen is even the resulting window length is is implicitly
            # rounded to next odd integer
            i_min = max(peak - wlen // 2, i_min)
            i_max = min(peak + wlen // 2, i_max)
            
        # Find the left base in interval [i_min, peak]
        i = left_bases[peak_nr] = peak
        left_min = amplitudes[peak]
        while i_min <= i and amplitudes[i] <= amplitudes[peak]:
            if amplitudes[i] < left_min:
                left_min = amplitudes[i]
                left_bases[peak_nr] = i
            i -= 1
                
        # Find the right base in interval [peak, i_max]
        i = right_bases[peak_nr] = peak
        right_min = amplitudes[peak]
        while i <= i_max and amplitudes[i] <= amplitudes[peak]:
            if amplitudes[i] < right_min:
                right_min = amplitudes[i]
                right_bases[peak_nr] = i
            i += 1
        
        prominences[peak_nr] = amplitudes[peak] - max(left_min, right_min)
        if prominences[peak_nr] == 0:
            show_warning = True
                
    if show_warning:
        warnings.warn("some peaks have a prominence of 0", PeakPropertyWarning, stacklevel=2)
    
    return prominences, left_bases, right_bases


### _peak_widths ###

def _peak_widths(x, peaks, rel_height, prominences, left_bases, right_bases):
    """
    Calcula la anchura de cada pico en la señal.
    Parametros
    ----------
    x : list
        Lista de valores con picos.
    peaks : list
        Indices de los picos en `x`.
    rel_height : np.float64
        Altura relativa a la cual se mide la altura del pico como
        porcentaje de su prominencia (ver `peak_widths`).
    prominences : ndarray
        Prominencias de cada pico en `peaks` tal y como fueron calculadas en `_peak_prominences`.
    left_bases, right_bases : ndarray
        Bases izquierdas y derechas de cada pico en `peaks` tal y como fueron calculadas en `_peak_prominences`.
    
    Salida
    -------
    widths : ndarray
        Anchuras de cada pico en número de muestras.
    width_heights : ndarray
        Altura de las líneas de contorno desde la cual se calcularon las anchuras `widths`.
        Esta altura es calculada como un porcentaje de la prominencia del pico.
    left_ips, right_ips : ndarray
        Posiciones interpoladas de los puntos de interesección izquierdo y derecho
        con la línea horizontal respectiva a la altura.
        
    Raises
    ------
    ValueError
        Si la `rel_height` es menor de 0.
        O si `peaks`, `left_bases` and `right_bases` no tienen las mismas dimensiones.
        O si los datos de `prominences` no cumplen la condición 
        ``0 <= left_base <= peak <= right_base < x.shape[0]`` para cada pico.

    Warnings
    --------
    PeakPropertyWarning
        Si la anchura calculada de algún pico es 0.
    """
    if rel_height < 0:
        raise ValueError('`rel_height` must be greater or equal to 0.0')
    if not (len(peaks) == prominences.shape[0] == left_bases.shape[0]
            == right_bases.shape[0]):
        raise ValueError("arrays in `prominence_data` must have the same shape as `peaks`")

    show_warning = False
    widths = np.empty(len(peaks), dtype=np.float64)
    width_heights = np.empty(len(peaks), dtype=np.float64)
    left_ips = np.empty(len(peaks), dtype=np.float64)
    right_ips = np.empty(len(peaks), dtype=np.float64)


    for p in range(len(peaks)):
        i_min = left_bases[p]
        i_max = right_bases[p]
        peak = peaks[p]
        # Validate bounds and order
        if not 0 <= i_min <= peak <= i_max < x.shape[0]:
            raise ValueError("prominence data is invalid for peak {}".format(peak))
        height = width_heights[p] = x[peak] - prominences[p] * rel_height

        # Find intersection point on left side
        i = peak
        while i_min < i and height < x[i]:
            i -= 1
        left_ip = i
        if x[i] < height:
            # Interpolate if true intersection height is between samples
            left_ip += (height - x[i]) / (x[i + 1] - x[i])

        # Find intersection point on right side
        i = peak
        while i < i_max and height < x[i]:
            i += 1
        right_ip = i
        if  x[i] < height:
            # Interpolate if true intersection height is between samples
            right_ip -= (height - x[i]) / (x[i - 1] - x[i])

        widths[p] = right_ip - left_ip
        if widths[p] == 0:
            show_warning = True
        left_ips[p] = left_ip
        right_ips[p] = right_ip

    if show_warning:
        warnings.warn("some peaks have a width of 0", PeakPropertyWarning, stacklevel=2)
    
    return widths, width_heights, left_ips, right_ips


### peak_estimator ###

from numpy import sqrt, sin, pi, sign

def peak_estimator(x, y, win="rect"):
    """
    This function estimates the position of the peak using interpolation.

    Parameters
    ----------
    x : list
        x values of a detected peak.
    y : list
        y values of a detected peak.
    win : string, optional
        This is the window used (for example "hann" or "rect"). 
        The default is "rect".

    Raises
    ------
    ValueError
        Index out of limits.

    Returns
    -------
    list
        [x,y] position of the new peak.

    """
    def corr_rect(deltaY):
        deltaF = 1 / (1 + deltaY)
        piDeltaF = pi*deltaF
        a = sin(piDeltaF)/piDeltaF
        return [deltaF, abs(a)]

    def corr_hann(deltaY):
        """Grandke estimator"""
        deltaF = (2 - deltaY) / (1 + deltaY)
        
        piDeltaF = pi*deltaF
        a = sin(piDeltaF)/piDeltaF
        b = 1/(1 - deltaF**2)
        return [deltaF, abs(a*b)]
    
    k0 = np.argmax(y)
    if k0 < 1 or k0 > len(y) - 2:
        raise ValueError("Index out of limits")
        
    # obtain neighbour points
    ns = (y[k0 - 1], y[k0 + 1])

    # line spacing (Hz)
    deltaF = x[2] - x[1]

    # var deltaY = ys[k0] / Math.max(prev, next);
    deltaY = y[k0] / max(ns[0], ns[1])

    if win == "rect":
        deltas = corr_rect(deltaY)
    elif win == "hann":
        deltas = corr_hann(deltaY)
    else:
        raise ValueError("Invalid window type")

    return [x[k0] + deltas[0]*deltaF*sign(ns[1] - ns[0]), y[k0]/deltas[1]]


### intersecccion ###

def interseccion(intervals):
    '''
    Funcion que sirve para encontrar la interseccion de varios intervalos.
    Argumentos de entrada
    ----------
    intervals: una lista de listas (intervalos definidos por su cota inf y su cota sup)
        Ejemplo: intervals = [[10, 110],[20, 60],[25, 75]]
    
    Valores de salida
    ----------
    [start, end] o [] (si la interseccion es vacia)
    '''
    start, end = intervals.pop() # extrae el ultimo elemento de la lista de intervalos

    while intervals:
        start_temp, end_temp = intervals.pop() # extrae el siguiente ultimo elemento de la lista de intervalos
        start = max(start, start_temp)         # compara los valores con start y end con los extremos del ultimo intervalo
        end = min(end, end_temp)
        if (start > end):                      # si el valor start es mayor que end, entonces el intervalo es vacío
            return []
        elif (start <= end):
            return [start, end]
        

### intervalosarmonicos ###

def intervalosarmonicos(i, v, Dv, orden_maximo):
    """
    Calcula los intervalos de incertidumbre de cada múltiplo r = 2,..., orden_maximo,
    para una componente espectral dada i, de i = [0 : F-1]
    Parametros
    ----------
    i: int
        Índice de la componente en la lista de componentes espectrales.
    v: list
        Frecuencias de las componentes espectrales.
    Dv: list
        Incertidumbre de las frecuencias de las componentes espectrales.
    orden_maximo: int
        Maximo orden con el cual se calcula una frecuencia de la serie.
    
    Valores de salida
    ----------
    intervalos: lista de listas
        Intervalos de incertidumbre (expresados en forma de lista, 
        por ejemplo: [0.51 , 0.85]) para cada frecuencia múltiplo.
    """
    intervalos = []
    
    for r in range(2, orden_maximo):
        r_liminf = r * (v[i] - (Dv[i]/2))
        r_limsup = r * (v[i] + (Dv[i]/2))
        intervalos.append( [r_liminf, r_limsup] )
        
    return intervalos


### encontrar_armonico ###

def encontrar_armonico(r, i, v, Dv, intervalo_componente, orden_maximo):
    """
    Funcion que encuentra el armónico de orden r para la componente espectral i.
    
    Parametros de entrada
    ----------
    r: int
        Orden del armónico. Valor por el que se multiplica la frecuencia fundamental de la componente i,
        encontrando así la frecuencia del armónico que se quiere buscar en el espectro.
    i: int
        Indice de la lista de componentes espectrales.
    v: list
        Frecuencias de las componentes espectrales.
    Dv: list
        Incertidumbre de las frecuencias de las componentes espectrales.
    intervalo_componente: list of lists
        Intervalos de incertidumbre para cada componente espectral i.
    orden_maximo: int
        Maximo orden con el cual se calcula una frecuencia de la serie.
    
    Valores de salida
    ----------
    armonico: float
        Frecuencia del armónico encontrado. Si no encuentra armónico, el valor de ese armónico es 0.
    indice: int
        Indice de la lista de componentes espectrales del armónico encontrado.
    
    Raises
    ------
    ValueError
        Si hay más de dos intervalos de incertidumbre de las componentes espectrales que cumplen que
        su intersección con el intervalo de incertidumbre del armónico de oden r de la componente i, es no vacía.
    """
    
    c = 0
    alista = []
    blista = []
    
    # actualizamos los intervalos de incertidumbre todos los armónicos de i para el nuevo v[i] y Dv[i]
    rxvi = intervalosarmonicos(i, v, Dv, orden_maximo)
    
    for j, componente in enumerate(intervalo_componente):  # para cada componente
        
        # calculamos la interseccion del intervalo de incertidumbre de la componente j con
        # el intervalo de incertidumbre del armónico de orden r de la componente i (rxvi[r-2]). 
        z = interseccion([componente, rxvi[r-2]])
        
        # Si el intervalo es no vacío:
        if z:
            # contabilizamos 1 intersección no vacía
            c += 1
            # añadimos a las listas los extremos del intervalo interseccion
            alista.append(z[0]) 
            blista.append(z[1])
            # damos el valor correpondiente al indice encontrado con interseccion no vacia
            indice = j
                    
            a, b = alista[0], blista[0]

    # Si se encuentran 2 intervalos que cumplen que la interseccion con rxvi[r-2] es no vacía
    if c == 2:
        # calcular cual de las dos frecuencias encontradas está más cerca de r x frecuencia componente i
        if abs(v[indice-1] - r*v[i]) < abs(v[indice] - r*v[i]):
            # actualizamos a, b e indice en consecuencia
            a = alista[0]
            b = blista[0]
            indice -= 1
        else:
            a, b = alista[1], blista[1]
    # Si hay más de 2 intervalos que lo cumplen: mostramos un error.
    elif c > 2:
        raise ValueError("Hay más de 2 intervalos que tienen intersección no vacía. Este algoritmo está diseñado para máximo 2 intervalos.")
        # Posibles mejoras: desarrollar esta parte de la función para más de 2 intervalos con intersección no vacía.
    
    if not c: # Si no se ha encontrado ningún armónico para ese orden r
        armonico = 0
        indice = None

    else: # Si sí se ha encontrado armónico
        armonico = v[indice]
        a = float(a)
        b = float(b)
        # actualizamos los v[i] y Dv[i]
        v[i] = (a+b) / (2*r)
        Dv[i] = (b-a) / r

    return armonico, indice


### encontrar_seriearmonica ###

def encontrar_seriearmonica(i, v, Dv, intervalo_componente, orden_maximo, criterio_parada):
    """
    Calcula la serie armónica de una componente espectral dada i.
    Parametros
    ----------
    i: int
        Índice de la componente en la lista de componentes espectrales.
    v: list
        Frecuencias de las componentes espectrales.
    Dv: list
        Incertidumbre de las frecuencias de las componentes espectrales.
    intervalo_componente: list of lists
        Intervalos de incertidumbre para cada componente espectral i.
    orden_maximo: int
        Maximo orden con el cual se calcula una frecuencia de la serie.
    criterio_parada: int
        Número máximo de veces que no se encuentra el armónico deseado, tras el cual
        el algoritmo deja de iterar (deja de buscar armónicos en la serie).
    
    Valores de salida
    ----------
    serie_armonica: lista
        Lista de los armónicos encontrados para la componente i. Las posiciones de la lista
        que contienen un 0 significan que no se ha encontrado armónico para ese orden r
    serie_armonica_indices: lista
        Lista de índices de las frecuencias de la serie armónica encontrada.
    numero_armonicos: int
        Número de armónicos de la serie sin contar la frecuencia fundamental.
    """
    serie_armonica = [v[i]]
    serie_armonica_indices = [i]
    numero_armonicos = 1
    k = 0
    for r in range(2, orden_maximo):
        
        # Busca el armónico número r
        armonico, indice = encontrar_armonico(r, i, v, Dv, intervalo_componente, orden_maximo)
        if (armonico == 0): # Si no encuentra armónico
            k += 1
        else: # Si encuentra un armónico
            numero_armonicos += 1
            k = 0
            serie_armonica_indices.append(indice)
        
        # Si después de k veces no ha encontrado armónico, la serie para
        if (k == criterio_parada):
            serie_armonica = serie_armonica[0:r-k]
            break
        
        serie_armonica.append(armonico)
        serie_armonica[0] = v[i] # actualizamos el primer valor de la serie ya que ha sido modificado
    
    return serie_armonica, serie_armonica_indices, numero_armonicos