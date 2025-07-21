import cv2
import numpy as np


def fourier_transform(gray_img):
    """
    Applica la trasformata di Fourier a un'immagine in scala di grigi.
    Ritorna lo spettro spostato e le componenti.
    """
    dft = cv2.dft(np.float32(gray_img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude, phase = cv2.cartToPolar(dft_shift[:, :, 0], dft_shift[:, :, 1])
    return dft_shift, magnitude, phase

def AVG_magnitude(magnitude):
    """
    Calcola la media della magnitudine dello spettro di Fourier.
    """
    log_amplitude = np.log1p(magnitude)
    return log_amplitude, cv2.blur(log_amplitude, (3, 3))

def Polar_to_Cartesian(spectral_residual, phase, dft_shift):
    """
    Converte coordinate polari in cartesiane.
    """
    real_part, imag_part = cv2.polarToCart(np.exp(spectral_residual), phase)
    dft_combined = np.zeros_like(dft_shift)
    dft_combined[:, :, 0] = real_part
    dft_combined[:, :, 1] = imag_part
    return dft_combined
    
def postProssessSaliency(saliency_map):
    """
    Applica un filtro gaussiano e normalizza la mappa di salienza.
    """
    saliency_map = cv2.GaussianBlur(saliency_map, (9, 9), 2.5)
    saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX)
    return saliency_map.astype(np.uint8)

def StaticSaliencySpectralResidual(image):
    """
    Calcola la mappa di salienza usando il metodo del residuo spettrale.
    
    Args:
        image (np.ndarray): immagine BGR.
        
    Returns:
        saliency_map (np.ndarray): immagine in scala di grigi (0-255).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Trasformata di Fourier
    dft_shift, magnitude, phase = fourier_transform(gray)
    
    # Calcolo residuo spettrale
    log_amplitude, avg_log_amplitude = AVG_magnitude(magnitude)
    spectral_residual = log_amplitude - avg_log_amplitude

    # Converti in coordinate cartesiane
    dft_combined= Polar_to_Cartesian(spectral_residual, phase, dft_shift)
    
    # Trasformata inversa
    idft = cv2.idft(np.fft.ifftshift(dft_combined))
    saliency = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])

    # Smussa e normalizza
    saliency = postProssessSaliency(saliency)
    
    return saliency

#saliency_map = StaticSaliencySpectralResidual(cv2.imread('/home/giovanni/Uni/multimedia/progetto/Holopix50k/val/left/-La6_F00B0o-361c1hl-_left.jpg'))
#cv2.imshow('Saliency Map', saliency_map)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
