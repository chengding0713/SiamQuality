import numpy as np
import random
import torch
from torch.utils.data import Dataset
from scipy.interpolate import interp1d
from scipy.ndimage import filters

# Augmentation functions

def gaussian_noise(signal, scale=0.1):
    """Add Gaussian noise to a 1D signal."""
    if scale == 0:
        return signal
    noise = np.random.normal(scale=scale, size=signal.shape)
    return signal + noise

def random_resized_crop(signal, crop_ratio_range=(0.5, 1.0)):
    """Extract a random crop of the signal and resize it to the original length."""
    crop_ratio = random.uniform(*crop_ratio_range)
    crop_length = int(crop_ratio * len(signal))
    start = random.randint(0, len(signal) - crop_length)
    cropped_signal = signal[start: start + crop_length]

    # Interpolating the cropped signal back to original length
    x = np.linspace(0, crop_length - 1, crop_length, endpoint=True)
    x_new = np.linspace(0, crop_length - 1, len(signal), endpoint=True)
    f = interp1d(x, cropped_signal, kind="linear")
    return f(x_new)

def channel_resize(signal, factor=1.5):
    return signal * factor

def negation(signal):
    return -signal

def dynamic_time_warp(signal, sigma=0.2):
    time_points = np.arange(len(signal))
    warped_length = int(sigma * len(signal))
    
    # Randomly choose start and end indices to resize signal after warping
    start = random.randint(0, len(signal) - warped_length)
    warped_signal = signal[start: start + warped_length]
    
    x = np.linspace(0, warped_length - 1, warped_length, endpoint=True)
    x_new = np.linspace(0, warped_length - 1, len(signal), endpoint=True)
    f = interp1d(x, warped_signal, kind="linear")
    return f(x_new)

def down_sample(signal, factor=2):
    downsampled_signal = signal[::factor]
    
    x = np.arange(len(downsampled_signal))
    x_new = np.linspace(0, len(downsampled_signal) - 1, len(signal), endpoint=True)
    f = interp1d(x, downsampled_signal, kind="linear")
    return f(x_new)

def time_warp(signal):
    return np.roll(signal, shift=100)

def time_out(signal, timeout_length=100):
    start = np.random.randint(0, len(signal) - timeout_length)
    signal[start:start+timeout_length] = 0
    return signal

def baseline_wander(signal):
    # Simulating baseline wander as a low-frequency sinusoidal interference
    baseline = np.sin(np.linspace(0, 10, len(signal)))
    return signal + baseline

def powerline_noise(signal, frequency=60, sample_rate=1200, magnitude=0.05):
    # Simulating powerline interference as a sinusoidal noise
    time = np.arange(len(signal))
    interference = magnitude * np.sin(2 * np.pi * frequency * time / sample_rate)
    return signal + interference

def em_noise(signal):
    # Placeholder for electromagnetic noise
    # Here, a combination of Gaussian noise and sinusoidal noise is used as an example.
    # Electromagnetic interference can often be periodic.
    noise = gaussian_noise(signal)
    interference = powerline_noise(noise)
    return interference

def baseline_shift(signal, shift=0.5):
    return signal + shift

def gaussian_blur_1d(signal, sigma=2):
    signal = np.float32(signal)
    return filters.gaussian_filter1d(signal, sigma)

def normalize(signal):
    return (signal - np.mean(signal)) / np.std(signal)

def transpose(signal):
    return np.flip(signal)


AUGMENTATION_MAP = {
    "gaussian_noise": gaussian_noise,
    "random_resized_crop": random_resized_crop,
    "channel_resize": channel_resize,
    "negation": negation,
    "dynamic_time_warp": dynamic_time_warp,
    "down_sample": down_sample,
    "time_warp": time_warp,
    "time_out": time_out,
    "baseline_wander": baseline_wander,
    "powerline_noise": powerline_noise,
    "em_noise": em_noise,
    "baseline_shift": baseline_shift,
    "gaussian_blur_1d": gaussian_blur_1d,
    "normalize": normalize,
    "transpose": transpose
}


class AugmentedDataset(Dataset):
    def __init__(self, data, augmentation_type=None, specified_augmentations=None, swav = False):
        """
        Parameters:
        - augmentation_type: 'random', 'specified', or None
        - specified_augmentations: a list of augmentation functions if 'specified' type is chosen
        """
        self.data = data
        self.augmentation_type = augmentation_type
        self.all_augmentations = [gaussian_noise, random_resized_crop, channel_resize, negation, dynamic_time_warp,
                                  down_sample, time_warp, time_out, baseline_wander, powerline_noise, em_noise,
                                  baseline_shift, gaussian_blur_1d, normalize, transpose]
        self.specified_augmentations = specified_augmentations or []
        self.swav = swav

    def _apply_random_augmentation(self, signal):
        chosen_augmentation = random.choice(self.all_augmentations)
        return chosen_augmentation(signal)

    def _apply_specified_augmentation(self, signal):
        if self.specified_augmentations:
            chosen_augmentation_name = random.choice(self.specified_augmentations)
            chosen_augmentation_func = AUGMENTATION_MAP.get(chosen_augmentation_name)
            if chosen_augmentation_func:
                return chosen_augmentation_func(signal)
        return signal

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
    
        original_signal = self.data[idx]
        augmented_signal = original_signal.copy()
        
        if self.swav:
            signal_list = []
            signal_list.append(original_signal)

        if self.augmentation_type == 'random':
            augmented_signal = self._apply_random_augmentation(augmented_signal)
            if self.swav:
                signal_list.append(augmented_signal)
                signal_list.append(self._apply_random_augmentation(augmented_signal))
        elif self.augmentation_type == 'specified':
            augmented_signal = self._apply_specified_augmentation(augmented_signal)
            if self.swav:
                signal_list.append(augmented_signal)
                signal_list.append(self._apply_specified_augmentation(augmented_signal))
        
        if self.swav:
            signal_list = list(map(lambda x: torch.tensor(x.copy()), signal_list))
            return signal_list

        return torch.tensor(original_signal.copy()), torch.tensor(augmented_signal.copy())

# # Example usage:

# # For random augmentation
# dataset_random = SignalDataset("path_to_npy_file.npy", 'random')

# # For a specified set of augmentations
# specified_augs = [gaussian_noise, random_resized_crop]
# dataset_specified = SignalDataset("path_to_npy_file.npy", 'specified', specified_augs)
