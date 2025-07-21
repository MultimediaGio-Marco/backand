import cv2
import numpy as np

class SpectralResidualSaliency:
    def __init__(self, blur_kernel=(9, 9), blur_sigma=2.5):
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma

    def fourier_transform(self, gray_img: np.ndarray):
        """
        Applica la trasformata di Fourier a un'immagine in scala di grigi.
        Ritorna lo spettro spostato e le componenti.
        """
        dft = cv2.dft(np.float32(gray_img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude, phase = cv2.cartToPolar(dft_shift[:, :, 0], dft_shift[:, :, 1])
        return dft_shift, magnitude, phase

    def avg_magnitude(self, magnitude: np.ndarray):
        """
        Calcola la media della magnitudine dello spettro di Fourier.
        """
        log_amplitude = np.log1p(magnitude)
        avg_log_amplitude = cv2.blur(log_amplitude, (3, 3))
        return log_amplitude, avg_log_amplitude

    def polar_to_cartesian(self, spectral_residual: np.ndarray, phase: np.ndarray, dft_shift: np.ndarray):
        """
        Converte coordinate polari in cartesiane.
        """
        real_part, imag_part = cv2.polarToCart(np.exp(spectral_residual), phase)
        dft_combined = np.zeros_like(dft_shift)
        dft_combined[:, :, 0] = real_part
        dft_combined[:, :, 1] = imag_part
        return dft_combined

    def postprocess_saliency(self, saliency_map: np.ndarray):
        """
        Applica un filtro gaussiano e normalizza la mappa di salienza.
        """
        saliency_map = cv2.GaussianBlur(saliency_map, self.blur_kernel, self.blur_sigma)
        saliency_map = cv2.normalize(saliency_map, None, 0, 255, cv2.NORM_MINMAX)
        return saliency_map.astype(np.uint8)

    def compute(self, image: np.ndarray):
        """
        Calcola la mappa di salienza usando il metodo del residuo spettrale.

        Args:
            image (np.ndarray): immagine BGR.

        Returns:
            np.ndarray: immagine in scala di grigi (0-255) della mappa di salienza.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        dft_shift, magnitude, phase = self.fourier_transform(gray)
        log_amplitude, avg_log_amplitude = self.avg_magnitude(magnitude)
        spectral_residual = log_amplitude - avg_log_amplitude
        dft_combined = self.polar_to_cartesian(spectral_residual, phase, dft_shift)

        idft = cv2.idft(np.fft.ifftshift(dft_combined))
        saliency = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])

        saliency = self.postprocess_saliency(saliency)

        return saliency
