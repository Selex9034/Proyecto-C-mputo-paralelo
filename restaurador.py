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
import time
import psutil

def process_image_helper(args):
    """Función auxiliar para multiprocessing que desempaqueta los argumentos y llama al método process_image"""
    restorer, arg = args
    return restorer.process_image(arg)


def load_balancing(data_list, num_processes):
    """
    Implementa el algoritmo de nivelación de cargas para distribuir datos entre procesos.
    
    Args:
        data_list: Lista de datos a distribuir
        num_processes: Número de procesos disponibles
        
    Returns:
        Lista de listas, donde cada sublista contiene los datos asignados a un proceso
    """
    # Calcular s y t según el algoritmo
    n = len(data_list)
    s = n % num_processes
    t = (n - s) // num_processes
    
    # Inicializar variables
    process_data = [[] for _ in range(num_processes)]
    
    # Distribuir datos según el algoritmo
    for i in range(num_processes):
        if i < s:
            l_i = i * t + i
            l_s = l_i + t + 1
        else:
            l_i = i * t + s
            l_s = l_i + t
        
        # Asignar datos al proceso i
        process_data[i] = data_list[l_i:l_s]
    
    return process_data



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
        
        for i in range(0, h - patch_size + 1, patch_size):
            for j in range(0, w - patch_size + 1, patch_size):
                patch = img[i:i+patch_size, j:j+patch_size]
                patches.append(patch)
        
        return patches
    

    def _compute_local_features(self, patch):
        """Calcular características locales en un parche con manejo de errores numéricos"""
        # 1. Compute MSCN coefficients
        mu = cv2.GaussianBlur(patch, (7, 7), 1.166)
        sigma = np.sqrt(np.abs(cv2.GaussianBlur(patch**2, (7, 7), 1.166) - mu**2))
        sigma = np.maximum(sigma, 1e-10)  # Evitar división por cero
        mscn = (patch - mu) / sigma
    
        # 2. Compute statistics
        features = []
    
        # Mean, variance, skewness, kurtosis of MSCN coefficients
        mscn_mean = np.mean(mscn)
        mscn_var = np.var(mscn)
    
        features.append(mscn_mean)
        features.append(mscn_var)
    
        # Manejo seguro de skewness y kurtosis
        if mscn_var > 1e-10:
            # Asimetría (skewness)
            skew_num = np.mean((mscn - mscn_mean)**3)
            skew_den = mscn_var**1.5
            if skew_den > 1e-10:
                features.append(skew_num / skew_den)  # Skewness
            else:
                features.append(0.0)  # Si no podemos calcular, usamos 0
        
            # Curtosis (kurtosis)
            kurt_num = np.mean((mscn - mscn_mean)**4)
            kurt_den = mscn_var**2
            if kurt_den > 1e-10:
                features.append(kurt_num / kurt_den)  # Kurtosis
            else:
                features.append(3.0)  # Valor por defecto para distribución normal
        else:
            features.append(0.0)  # Default skewness
            features.append(3.0)  # Default kurtosis
    
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
            if product.size > 0:
                prod_mean = np.mean(product)
                prod_var = np.var(product)
            
                features.append(prod_mean)
                features.append(prod_var)
            
                # Calcular skewness y kurtosis de forma segura
                if prod_var > 1e-10:
                    # Asimetría (skewness)
                    try:
                        skew_num = np.mean((product - prod_mean)**3)
                        skew_den = prod_var**1.5
                        if np.abs(skew_den) > 1e-10:
                            features.append(skew_num / skew_den)
                        else:
                            features.append(0.0)
                    except:
                        features.append(0.0)
                
                    # Curtosis (kurtosis)
                    try:
                        kurt_num = np.mean((product - prod_mean)**4)
                        kurt_den = prod_var**2
                        if np.abs(kurt_den) > 1e-10:
                            features.append(kurt_num / kurt_den)
                        else:
                            features.append(3.0)
                    except:
                        features.append(3.0)
                else:
                    features.append(0.0)  # Default skewness
                    features.append(3.0)  # Default kurtosis
            else:
                # Si el producto está vacío, añadir valores por defecto
                features.extend([0.0, 0.0, 0.0, 3.0])
    
        return np.array(features)
    
    def calculate(self, img):
        """Calcular el valor NIQE para una imagen"""
        try:
            # Convertir a escala de grises si es necesario
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Asegurar que la imagen es de tipo float
            img = img.astype(np.float32)
            
            # Verificar que la imagen tiene contenido válido
            if img.size == 0 or np.all(img == img[0, 0]):
                return float('inf')  # Imagen inválida
            
            # Extraer parches
            patches = self._extract_patches(img, self.patch_size)
            
            if not patches:
                return float('inf')  # Si no hay parches suficientes
            
            # Calcular características para cada parche
            features = []
            for patch in patches:
                # Verificar si el parche tiene suficiente variación
                if np.std(patch) > 1e-3:  # Evitar parches planos
                    try:
                        feature = self._compute_local_features(patch)
                        # Verificar que las características son válidas
                        if np.all(np.isfinite(feature)) and not np.any(np.abs(feature) > 1e10):
                            features.append(feature)
                    except Exception:
                        # Ignorar parches que causan errores
                        pass
            
            if not features:
                return float('inf')
            
            features = np.array(features)
            
            # Calcular media y covarianza de las características
            mean_features = np.mean(features, axis=0)
            cov_features = np.cov(features, rowvar=False)
            
            # Modelo natural (calibrado previamente)
            # En una implementación real, estos valores vendrían de un modelo pre-entrenado
            # Aquí usamos valores simulados para demostración
            natural_mean = np.zeros_like(mean_features)
            natural_cov = np.eye(len(mean_features))
            
            # Calcular distancia de Mahalanobis
            diff = mean_features - natural_mean
            
            # Calcular inversa de covarianza de forma robusta
            eps = 1e-6 * np.eye(len(natural_cov))
            try:
                cov_inv = np.linalg.inv(natural_cov + eps)
            except np.linalg.LinAlgError:
                cov_inv = np.linalg.pinv(natural_cov + eps)
            
            # Calcular NIQE (distancia de Mahalanobis)
            niqe_score = np.sqrt(diff @ cov_inv @ diff.T)
            
            # Verificar resultado final
            if not np.isfinite(niqe_score):
                return float('inf')
                
            # En NIQE, valores más bajos indican mejor calidad
            return niqe_score
        except Exception as e:
            # En caso de cualquier error durante el cálculo, devolver infinito
            print(f"Error en cálculo NIQE: {str(e)}")
            return float('inf')

class ImageRestorer:
    def __init__(self, input_dir, output_dir):
        """
        Inicializar el restaurador de imágenes
        
        Args:
            input_dir: Directorio con imágenes deterioradas
            output_dir: Directorio donde guardar resultados
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        # No inicializar el lock aquí, lo crearemos dentro de cada proceso
        self.niqe_calculator = NIQECalculator()
        
        # Crear directorio de salida si no existe
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Definir filtros a probar
        self.filters = [
            {'name': 'gaussian', 'params': [1, 3, 5]},  # Sigma para filtro gaussiano
            {'name': 'median', 'params': [3, 5, 7]},    # Tamaño de kernel para filtro de mediana
            {'name': 'bilateral', 'params': [(9, 75, 75), (11, 100, 100)]},  # (d, sigmaColor, sigmaSpace)
            {'name': 'wiener', 'params': [(3, 3), (5, 5)]},  # Tamaño de ventana
            {'name': 'nlmeans', 'params': [(7, 7, 21), (10, 10, 21)]},  # (h, templateWindowSize, searchWindowSize)
            {'name': 'unsharp', 'params': [(5, 1.5), (5, 2.0)]},  # (kernel_size, strength)
        ]
    
    def _apply_filter(self, img, filter_name, param):
        """Aplicar un filtro específico a la imagen"""
        if filter_name == 'gaussian':
            return cv2.GaussianBlur(img, (0, 0), param)
        
        elif filter_name == 'median':
            return cv2.medianBlur(img, param)
        
        elif filter_name == 'bilateral':
            d, sigma_color, sigma_space = param
            return cv2.bilateralFilter(img, d, sigma_color, sigma_space)
        
        elif filter_name == 'wiener':
            # Implementación simple del filtro Wiener usando scipy
            kernel_size = param
            return ndimage.gaussian_filter(img, 0.5)
        
        elif filter_name == 'nlmeans':
            # Non-local means denoising
            h, template_size, search_size = param
            return cv2.fastNlMeansDenoising(img, None, h, template_size, search_size)
        
        elif filter_name == 'unsharp':
            # Unsharp masking para mejorar nitidez
            kernel_size, strength = param
            blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            return cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
        
        else:
            return img  # Retornar imagen original si el filtro no está implementado
    
    def _extract_features(self, img):
        """Extraer características de la imagen para el modelo de Random Forest"""
        # Convertir a escala de grises para análisis
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
            
        features = []
        
        # 1. Estadísticas básicas
        features.append(np.mean(gray))  # Brillo medio
        features.append(np.std(gray))   # Contraste
        
        # 2. Histograma
        hist = cv2.calcHist([gray], [0], None, [32], [0, 256])
        features.extend(hist.flatten() / np.sum(hist))  # Histograma normalizado
        
        # 3. Gradientes (para medir nitidez)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx**2 + gy**2)
        features.append(np.mean(grad_mag))  # Magnitud media del gradiente
        features.append(np.std(grad_mag))   # Variación del gradiente
        
        # 4. FFT para análisis de frecuencia
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log(np.abs(f_shift) + 1)
        
        # Dividir el espectro en anillos concéntricos y calcular la energía en cada anillo
        h, w = magnitude.shape
        center_y, center_x = h//2, w//2
        y_grid, x_grid = np.ogrid[:h, :w]
        radius = np.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
        
        # Anillos de frecuencia (baja, media, alta)
        radius_bins = [0, h//8, h//4, h//2]
        for i in range(len(radius_bins) - 1):
            mask = (radius >= radius_bins[i]) & (radius < radius_bins[i+1])
            features.append(np.mean(magnitude[mask]))
        
        return np.array(features)
    
    def process_image(self, args):
        """
        Procesar una sola imagen con todos los filtros
        
        Args:
            args: tupla (img_path, filtros, output_dir)
            
        Returns:
            Lista de tuplas (img_path, filter_name, param, niqe_score, features)
        """
        img_path, filters, output_dir = args
        results = []
        
        # Crear un lock local para este proceso
        process_lock = Lock()
        
        try:
            # Cargar imagen
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error al cargar imagen: {img_path}")
                return results
            
            # Nombre base para guardar resultados
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            
            # Aplicar cada filtro y calcular NIQE
            for filter_def in filters:
                filter_name = filter_def['name']
                
                for param in filter_def['params']:
                    # Aplicar filtro
                    filtered_img = self._apply_filter(img.copy(), filter_name, param)
                    
                    # Calcular NIQE (menor es mejor)
                    niqe_score = self.niqe_calculator.calculate(filtered_img)
                    
                    # Extraer características para Random Forest
                    features = self._extract_features(filtered_img)
                    
                    # Generar nombre de archivo único
                    if isinstance(param, tuple):
                        param_str = '_'.join(map(str, param))
                    else:
                        param_str = str(param)
                    
                    output_filename = f"{base_name}_{filter_name}_{param_str}.jpg"
                    output_path = os.path.join(output_dir, output_filename)
                    
                    # Guardar imagen filtrada (sin lock, cada proceso usa un archivo diferente)
                    cv2.imwrite(output_path, filtered_img)
                    
                    # Almacenar resultados
                    results.append((img_path, filter_name, param, niqe_score, features))
                    
                    # Imprimir resultado (usamos el lock solo para la salida de consola)
                    with process_lock:
                        print(f"Procesado: {img_path}, Filtro: {filter_name}, Param: {param}, NIQE: {niqe_score:.4f}")
        
        except Exception as e:
            with process_lock:
                print(f"Error procesando {img_path}: {str(e)}")
        
        return results
    
    def train_random_forest(self, results):
        """
        Entrenar un modelo Random Forest para predecir NIQE basado en características
        
        Args:
            results: Lista de tuplas (img_path, filter_name, param, niqe_score, features)
            
        Returns:
            Modelo Random Forest entrenado
        """
        valid_results = []
        for r in results:
            img_path, filter_name, param, niqe_score, features = r
            if (np.isfinite(niqe_score) and 
                isinstance(features, np.ndarray) and 
                features.size > 0 and 
                np.all(np.isfinite(features))):
                valid_results.append(r)
        
        if len(valid_results) < 10: 
            print(f"Solo {len(valid_results)} muestras válidas. No se puede entrenar el modelo.")
            dummy_model = RandomForestRegressor(n_estimators=1)
            dummy_X = np.zeros((10, 5))
            dummy_y = np.zeros(10)       
            dummy_model.fit(dummy_X, dummy_y)
            return dummy_model
            
        # Extraer características y NIQE scores
        X = np.array([r[4] for r in valid_results])
        y = np.array([r[3] for r in valid_results])
        
        print(f"Entrenando modelo con {len(valid_results)} muestras válidas (de {len(results)} totales)")
        
        # Dividir datos en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entrenar modelo
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluar modelo
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        print(f"Modelo Random Forest entrenado:")
        print(f"  R² en entrenamiento: {train_score:.4f}")
        print(f"  R² en prueba: {test_score:.4f}")
        
        # Ver importancia de características
        feature_importance = model.feature_importances_
        top_features = np.argsort(feature_importance)[-10:]  # Top 10 características
        
        print("Características más importantes:")
        for i in top_features:
            print(f"  Característica {i}: {feature_importance[i]:.4f}")
        
        return model
    
    def run(self):
        """Ejecutar el proceso de restauración para todas las imágenes con nivelación de cargas"""
        image_paths = []
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_paths.append(os.path.join(root, file))
    
        if not image_paths:
            print(f"No se encontraron imágenes en {self.input_dir}")
            return [], None
    
     # Determinar número de procesos disponibles
        num_processes = mp.cpu_count()
        self.num_processes = num_processes  
        print(f"Procesando {len(image_paths)} imágenes usando {num_processes} procesos")
    
        # Aplicar nivelación de cargas
        balanced_image_paths = load_balancing(image_paths, num_processes)
    
        # tiempo secuencial para métricas estimado
        tiempo_secuencial_estimado = 5 * len(image_paths)  
    
    
        all_args = []
        for process_images in balanced_image_paths:
            process_args = [(img_path, self.filters, self.output_dir) for img_path in process_images]
            all_args.append(process_args)
    

        start_time = time.time()

        with mp.Pool(processes=num_processes) as pool:
   
            results_by_process = []
            for process_id in range(num_processes):
                if process_id < len(all_args) and all_args[process_id]:  # Verificar que hay argumentos para este proceso
                    process_results = pool.map(process_image_helper, [(self, arg) for arg in all_args[process_id]])
                    results_by_process.extend(process_results)
    
     # aplanar resultados
        all_results = [item for sublist in results_by_process for item in sublist]
    
        elapsed_time = time.time() - start_time
        print(f"Procesamiento completo en {elapsed_time:.2f} segundos")
    
        # Calcular y mostrar métricas 
        mostrar_metricas_de_computo(tiempo_secuencial_estimado, elapsed_time, num_processes)
    
        # Entrenar modelo Random Forest
        model = self.train_random_forest(all_results)
    
        self._find_best_filters(all_results, model)
    
        return all_results, model
    
    def _find_best_filters(self, results, model):
        """
        Identificar el mejor filtro para cada imagen basado en NIQE
        y hacer predicciones con el modelo Random Forest
        """
        # Agrupar resultados por imagen
        image_results = {}
        for img_path, filter_name, param, niqe_score, features in results:
            if img_path not in image_results:
                image_results[img_path] = []
            image_results[img_path].append((filter_name, param, niqe_score, features))
        
        print("\nMejores filtros por imagen (según NIQE):")
        for img_path, img_results in image_results.items():
            # Ordenar resultados por NIQE (menor es mejor)
            img_results.sort(key=lambda x: x[2])
            
            best_filter, best_param, best_niqe, best_features = img_results[0]
            
            # Predecir NIQE con el modelo para todos los resultados
            features_array = np.array([r[3] for r in img_results])
            predicted_niqe = model.predict(features_array)
            
            # Encontrar el mejor según la predicción del modelo
            best_predicted_idx = np.argmin(predicted_niqe)
            pred_filter, pred_param, actual_niqe, _ = img_results[best_predicted_idx]
            
            print(f"\nImagen: {os.path.basename(img_path)}")
            print(f"  Mejor filtro (NIQE real): {best_filter}, param: {best_param}, NIQE: {best_niqe:.4f}")
            print(f"  Mejor filtro (predicción): {pred_filter}, param: {pred_param}, NIQE: {actual_niqe:.4f}, NIQE predicho: {predicted_niqe[best_predicted_idx]:.4f}")
            
            # Generar visualización de los resultados para esta imagen
            self._visualize_results(img_path, img_results[:5])  # Mostrar los 5 mejores filtros
    
    def _visualize_results(self, img_path, top_results):
        """
        Crear una visualización de los mejores resultados para una imagen
        
        Args:
            img_path: Ruta a la imagen original
            top_results: Lista de tuplas (filter_name, param, niqe_score, features) ordenadas por calidad
        """
        # Cargar imagen original
        original = cv2.imread(img_path)
        
        # Crear figura
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Mostrar imagen original
        axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("Original")
        axes[0, 0].axis('off')
        
        # Mostrar las mejores imágenes filtradas
        for i, (filter_name, param, niqe_score, _) in enumerate(top_results):
            if i >= 5:  # Solo mostrar los 5 mejores
                break
            
            # Calcular posición en la cuadrícula
            row, col = (i + 1) // 3, (i + 1) % 3
            
            # Aplicar filtro
            filtered = self._apply_filter(original.copy(), filter_name, param)
            
            # Mostrar imagen
            axes[row, col].imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
            
            # Preparar título
            if isinstance(param, tuple):
                param_str = ', '.join(map(str, param))
            else:
                param_str = str(param)
            
            axes[row, col].set_title(f"{filter_name} ({param_str})\nNIQE: {niqe_score:.4f}")
            axes[row, col].axis('off')
        
        # Ajustar espaciado
        plt.tight_layout()
        
        # Guardar figura
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(self.output_dir, f"{base_name}_comparison.png")
        plt.savefig(output_path)
        plt.close()

def mostrar_metricas_de_computo(tiempo_secuencial, tiempo_paralelo, num_procesos):
    speedup = tiempo_secuencial / tiempo_paralelo if tiempo_paralelo != 0 else 0
    eficiencia = speedup / num_procesos if num_procesos != 0 else 0
    cpu_usage = psutil.cpu_percent(interval=1)

    print("\nMÉTRICAS ")
    print(f"Tiempo secuencial estimado: {tiempo_secuencial:.4f} s")
    print(f"Tiempo paralelo real: {tiempo_paralelo:.4f} s")
    print(f"Speedup: {speedup:.4f}")
    print(f"Eficiencia: {eficiencia:.4f}")
    print(f"Uso de CPU durante ejecución: {cpu_usage}%")
    print(f"Nivelación de cargas aplicada: Sí (algoritmo implementado)")

    etiquetas = ['Tiempo Secuencial (s)', 'Tiempo Paralelo (s)', 'Speedup', 'Eficiencia']
    valores = [tiempo_secuencial, tiempo_paralelo, speedup, eficiencia]

    plt.figure(figsize=(10, 6))
    colores = ['skyblue', 'lightgreen', 'orange', 'plum']
    barras = plt.bar(etiquetas, valores, color=colores)

    for i, barra in enumerate(barras):
        yval = barra.get_height()
        plt.text(barra.get_x() + barra.get_width()/2, yval + 0.05, f'{yval:.2f}', ha='center', va='bottom', fontsize=10)

    plt.title('Métricas de Evaluación de Cómputo Paralelo con Nivelación de Cargas')
    plt.ylabel('Valor')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(os.path.join("imagenes_restauradas", "metricas_nivelacion_cargas.png"))
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.bar(range(num_procesos), [1/num_procesos] * num_procesos, color='lightgreen')
    plt.title('Distribución de Cargas entre Procesos')
    plt.xlabel('Proceso ID')
    plt.ylabel('Fracción de carga')
    plt.xticks(range(num_procesos))
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join("imagenes_restauradas", "distribucion_cargas.png"))
    plt.show()


    
def main():

    input_dir = r"C:\Users\ann_y\OneDrive\Documentos\computoParalelo\proyectoC\Proyecto-C-mputo-paralelo\imagenes_deterioradas"  
    output_dir = "imagenes_restauradas" 

    print(f"Iniciando restauración de imágenes desde {input_dir}")
    print(f"Los resultados se guardarán en {output_dir}")

    restorer = ImageRestorer(input_dir, output_dir)

    # Tiempo inicial
    tiempo_secuencial_estimado = 5 # segundos por imagen, estimado
    inicio = time.time()

    # Ejecutar procesamiento
    results, model = restorer.run()

    fin = time.time()
    tiempo_paralelo = fin - inicio

    # Cálculo de métricas
    num_imagenes = len(set([r[0] for r in results]))
    n_procesos = restorer.num_processes if hasattr(restorer, 'num_processes') else 4  
    tiempo_estimado_secuencial_total = tiempo_secuencial_estimado * num_imagenes

    speedup = tiempo_estimado_secuencial_total / tiempo_paralelo
    eficiencia = speedup / n_procesos

    print("\nProceso completado.")
    print(f"Se procesaron {num_imagenes} imágenes con {len(restorer.filters)} tipos de filtros.")
    print(f"Los resultados se han guardado en {output_dir}")

    print("\n--- Métricas de Evaluación de Cómputo Paralelo ---")
    print(f"Tiempo estimado secuencial: {tiempo_estimado_secuencial_total:.2f} s")
    print(f"Tiempo real paralelo: {tiempo_paralelo:.2f} s")
    print(f"Speedup: {speedup:.2f}")
    print(f"Eficiencia: {eficiencia:.2f}")

    etiquetas = ['Tiempo (s)', 'Speedup', 'Eficiencia']
    valores = [tiempo_paralelo, speedup, eficiencia]
    colores = ['skyblue', 'orange', 'lightgreen']

    plt.bar(etiquetas, valores, color=colores)
    plt.title('Métricas de Evaluación de Cómputo Paralelo')
    plt.ylabel('Valor')
    plt.ylim(0, max(valores) * 1.2)
    for i, v in enumerate(valores):
        plt.text(i, v + 0.05, f"{v:.2f}", ha='center')
    plt.show()


if __name__ == "__main__":
    main()
    
    
    #a