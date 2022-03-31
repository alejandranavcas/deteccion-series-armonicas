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
        
