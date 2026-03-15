import numpy as np
import zipfile
import noise

from scipy import ndimage

def load_lesion(zip_file, target_size, vxl):
    
    # Convert target_size from mm to voxel space
    target = np.array(target_size) / np.array(vxl)
    target = np.round(target).astype(int) 

    with zipfile.ZipFile(zip_file, 'r') as zip_ref:

        file = [f for f in zip_ref.namelist() if f.endswith('.raw')][0]
        size = str(file).replace('.raw', '').split('_')[-1].split('x')

        with zip_ref.open(file) as f:
            map = f.read()
            map = np.frombuffer(bytearray(map), dtype=np.uint8).reshape(int(size[2]), int(size[1]), int(size[0]))
    
    # Resize the lesion mask to match the target size in voxels
    scaling_factors = target / np.array(map.shape)  # scaling per dimension
    resized_lesion = ndimage.zoom(map.astype(float), scaling_factors, order=1) #> 0.5
    resized_lesion = resized_lesion/np.max(resized_lesion) # Normalize to [0, 1]
    
    return resized_lesion.astype(np.float32)

def perlin_noise_3d(shape, scale=10, seed=None):
    if seed is not None:
        np.random.seed(seed)
    noise_volume = np.zeros(shape, dtype=np.float32)
    
    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                noise_volume[x][y][z] = noise.pnoise3(
                    x / scale, y / scale, z / scale, octaves=3, persistence=0.5, 
                    lacunarity=2.0, repeatx=shape[0], repeaty=shape[1], repeatz=shape[2], base=seed
                )
    
    # Normalize noise to [0, 1]
    noise_volume = (noise_volume - noise_volume.min()) / (noise_volume.max() - noise_volume.min())
    return noise_volume

def modify_lesion(lesion3d):

    distance_map = ndimage.distance_transform_edt(lesion3d)
    distance_map_normalized = distance_map / distance_map.max()
    perlin_noise = perlin_noise_3d(lesion3d.shape, scale=20, seed=42)
    contrast_volume = (0.5 * distance_map_normalized + 0.5 * perlin_noise * lesion3d).astype(np.float32)
    contrast_volume = (contrast_volume - contrast_volume.min()) / (contrast_volume.max() - contrast_volume.min())

    return np.transpose(contrast_volume, (1, 0, 2))