import os
import numpy as np
import cv2
import multiprocessing as mp
from multiprocessing import Lock
import time
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy import ndimage
from scipy import signal
import matplotlib.pyplot as plt


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
    
    def _compute_local_features(self, patch):
        """Calcular características locales en un parche"""
        # 1. Compute MSCN coefficients
        mu = cv2.GaussianBlur(patch, (7, 7), 1.166)
        sigma = np.sqrt(np.abs(cv2.GaussianBlur(patch**2, (7, 7), 1.166) - mu**2))
        sigma = np.maximum(sigma, 1e-10)  # Evitar división por cero
        mscn = (patch - mu) / sigma
        
        # 2. Compute statistics
        features = []
        
        # Mean, variance, skewness, kurtosis of MSCN coefficients
        features.append(np.mean(mscn))
        features.append(np.var(mscn))
        features.append(np.mean((mscn - np.mean(mscn))**3) / (np.var(mscn)**1.5))  # Skewness
        features.append(np.mean((mscn - np.mean(mscn))**4) / (np.var(mscn)**2))    # Kurtosis
        
        # 3. Compute pairwise products
        h_shift = np.roll(mscn, 1, axis=1)
        v_shift = np.roll(mscn, 1, axis=0)
        d1_shift = np.roll(np.roll(mscn, 1, axis=0), 1, axis=1)
        d2_shift = np.roll(np.roll(mscn, 1, axis=0), -1, axis=1)
        
        # Productos de coeficientes adyacentes
        h_product = mscn[:, :-1] * mscn[:, 1:]
        v_product = mscn[:-1, :] * mscn[1:, :]
        d1_product = mscn[:-1, :-1] * mscn[1:, 1:]
        d2_product = mscn[:-1, 1:] * mscn[1:, :-1]
        
        # Estadísticas de los productos
        for product in [h_product.flatten(), v_product.flatten(), 
                        d1_product.flatten(), d2_product.flatten()]:
            features.append(np.mean(product))
            features.append(np.var(product))
            features.append(np.mean((product - np.mean(product))**3) / (np.var(product)**1.5))
            features.append(np.mean((product - np.mean(product))**4) / (np.var(product)**2))
        
        return np.array(features)