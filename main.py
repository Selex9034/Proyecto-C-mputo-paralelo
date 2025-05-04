import os
import numpy as np
import cv2
import multiprocessing as mp
from multiprocessing import Lock
import time
import random

# Función auxiliar para multiprocessing
def process_image_helper(args):
    """Función auxiliar para multiprocessing que desempaqueta los argumentos y llama al método process_image"""
    restorer, arg = args
    return restorer.process_image(arg)


class NIQECalculator:
    """Implementación de Non-reference Image Quality Evaluator (NIQE)"""
    
    def __init__(self):
        # Parámetros recomendados para NIQE
        self.patch_size = 96
        self.window = self._gaussian_window(7)
    
    def _gaussian_window(self, size, sigma=1.5):
        """Crear ventana gaussiana para el cálculo de características"""
        x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
        return g/g.sum()
    
    def _extract_patches(self, img, patch_size):
        """Extraer parches de la imagen para análisis local"""
        h, w = img.shape
        patches = []
        
        # Obtener parches no superpuestos
        for i in range(0, h - patch_size + 1, patch_size):
            for j in range(0, w - patch_size + 1, patch_size):
                patch = img[i:i+patch_size, j:j+patch_size]
                patches.append(patch)
        
        return patches