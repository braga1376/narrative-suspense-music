import numpy as np
from typing import List, Tuple

# ['C','D-','D','E-','E','F','F#','G','A-','A','B-','B']
NOTE_INDEX_TO_PITCH_INDEX = [0, -5, 2, -3, 4, -1, -6, 1, -4, 3, -2, 5]
RADIUS = 1.0
VERTICAL_STEP = 0.4
CHORD_WEIGHT = [0.536, 0.274, 0.19]
ALPHA = 0.75
BETA = 0.75

PITCH_POSITIONS = {
    0: np.array([0.0, RADIUS, 0.0]),
    1: np.array([RADIUS, 0.0, 0.0]),
    2: np.array([0.0, -RADIUS, 0.0]),
    3: np.array([-RADIUS, 0.0, 0.0])
}

def pitch_index_to_position(pitch_index: int) -> np.ndarray:
    """Convert a pitch index to a spiral array position"""
    c = pitch_index % 4
    pos = PITCH_POSITIONS[c].copy()
    pos[2] = pitch_index * VERTICAL_STEP
    return pos

def triad_position(root_index: int, third_offset: int, fifth_offset: int) -> np.ndarray:
    """Calculate the position of a triad given the root index and offsets for the third and fifth"""
    root_pos = pitch_index_to_position(root_index)
    third_pos = pitch_index_to_position(root_index + third_offset)
    fifth_pos = pitch_index_to_position(root_index + fifth_offset)
    centre_pos = CHORD_WEIGHT[0] * root_pos + CHORD_WEIGHT[1] * third_pos + CHORD_WEIGHT[2] * fifth_pos
    return centre_pos

def major_triad_position(root_index: int) -> np.ndarray:
    """Calculate the position of a major triad given the root index"""
    return triad_position(root_index, 4, 1)

def minor_triad_position(root_index: int) -> np.ndarray:
    """Calculate the position of a minor triad given the root index"""
    return triad_position(root_index, -3, 1)

def major_key_position(key_index: int) -> np.ndarray:
    """Calculate the position of a major key given the key index"""
    root_triad_pos = major_triad_position(key_index)
    fifth_triad_pos = major_triad_position(key_index + 1)
    fourth_triad_pos = major_triad_position(key_index - 1)
    key_pos = CHORD_WEIGHT[0] * root_triad_pos + CHORD_WEIGHT[1] * fifth_triad_pos + CHORD_WEIGHT[2] * fourth_triad_pos
    return key_pos

def minor_key_position(key_index: int) -> np.ndarray:
    """Calculate the position of a minor key given the key index"""
    root_triad_pos = minor_triad_position(key_index)
    major_fifth_triad_pos = major_triad_position(key_index + 1)
    minor_fifth_triad_pos = minor_triad_position(key_index + 1)
    major_fourth_triad_pos = major_triad_position(key_index - 1)
    minor_fourth_triad_pos = minor_triad_position(key_index - 1)
    key_pos = CHORD_WEIGHT[0] * root_triad_pos + \
              CHORD_WEIGHT[1] * (ALPHA * major_fifth_triad_pos + (1 - ALPHA) * minor_fifth_triad_pos) + \
              CHORD_WEIGHT[2] * (BETA * minor_fourth_triad_pos + (1 - ALPHA) * major_fourth_triad_pos)
    return key_pos

def notes_to_spiral_array(notes: List[int]) -> np.ndarray:
    """Convert a list of notes to a spiral array position"""
    return np.array([pitch_index_to_position(NOTE_INDEX_TO_PITCH_INDEX[(note) % 12]) for note in notes])

def calculate_center_of_effect(positions: np.ndarray, weights: List=None) -> np.ndarray:
    """Calculate the center of effect for a list of positions"""
    return np.average(positions, weights=weights)

def calculate_cloud_diameter(positions: np.ndarray) -> float:
    """Calculate the cloud diameter for a list of positions"""
    n = len(positions)
    if n < 2:
        return 0.0
    distances = np.linalg.norm(positions[:, np.newaxis] - positions, axis=2)
    return np.max(distances)

def calculate_cloud_momentum(positions1: np.ndarray, positions2: np.ndarray) -> float:
    """Calculate the cloud momentum between two lists of positions"""
    return np.linalg.norm(calculate_center_of_effect(positions1) - calculate_center_of_effect(positions2))

def calculate_tensile_strain(positions: np.ndarray, key_position: np.ndarray) -> float:
    """Calculate the tensile strain for a list of positions"""
    return np.linalg.norm(calculate_center_of_effect(positions) - calculate_center_of_effect(key_position))

def transpose_to_c(bars: List[List[int]], key_root: int, is_major: bool) -> List[List[int]]:
    """Transpose notes to the key of C"""
    transposition_interval = -key_root if is_major else -(key_root + 3)
    return [[(note + transposition_interval) % 12 for note in bar] for bar in bars]

def calculate_tension(bars: List[List[int]], key_root: int, is_major: bool) -> Tuple[List[float], List[float], List[float]]:
    """Calculate the tension for a list of notes"""
    transposed_bars = transpose_to_c(bars, key_root, is_major)
    key_root = 0 if is_major else 9

    positions_bars = [notes_to_spiral_array(transposed_notes) for transposed_notes in transposed_bars]
    key_position = major_key_position(key_root) if is_major else minor_key_position(key_root)

    cloud_diameter = [calculate_cloud_diameter(positions) for positions in positions_bars]

    positions_bars_shifted = positions_bars[1:]
    cloud_momentum = [0.0] + [calculate_cloud_momentum(positions_1, positions_2) for positions_1, positions_2 in zip(positions_bars[:-1], positions_bars_shifted)]

    tensile_strain = [calculate_tensile_strain(positions, key_position) for positions in positions_bars]

    return cloud_diameter, cloud_momentum, tensile_strain


def calculate_tension_fast(bars: List[List[int]], key_root: int, is_major: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate tension with handling for variable-length bars"""
    transposition_interval = -key_root if is_major else -(key_root + 3)
    transposed_notes = [[(note + transposition_interval) % 12 for note in bar] for bar in bars]
    
    positions_bars = [notes_to_spiral_array(notes) for notes in transposed_notes]
    
    key_root = 0 if is_major else 9
    key_position = major_key_position(key_root) if is_major else minor_key_position(key_root)
    
    n_bars = len(positions_bars)
    cloud_diameter = np.zeros(n_bars)
    cloud_momentum = np.zeros(n_bars)
    tensile_strain = np.zeros(n_bars)
    
    centers = np.array([np.mean(positions, axis=0) for positions in positions_bars])
    key_center = np.mean(key_position, axis=0)
    
    for i in range(n_bars):
        positions = positions_bars[i]
        
        # Cloud diameter
        if len(positions) >= 2:
            distances = np.linalg.norm(positions[:, np.newaxis] - positions, axis=2)
            cloud_diameter[i] = np.max(distances)
        
        # Tensile strain
        tensile_strain[i] = np.linalg.norm(centers[i] - key_center)
        
        # Cloud momentum
        if i > 0:
            cloud_momentum[i] = np.linalg.norm(centers[i] - centers[i-1])
    
    return cloud_diameter, cloud_momentum, tensile_strain