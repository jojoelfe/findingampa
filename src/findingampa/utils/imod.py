def get_exclude_list():
    exclude_list = ''
    # Open up align.com file and find the line starting with ExcludeList
    with open('align.com', 'r') as f:
        for line in f:
            if line.startswith('ExcludeList'):
                exclude_list = line.split()[-1:][0]
    
    return exclude_list

def get_align_angle_offset():
    # Open up align.com file and find the line starting with AlignAngle
    with open('tomopitch.log', 'r') as f:
        for line in f:
            if line.startswith(' Angle offset'):
                align_angle = line.split()[-1:][0]
    
    return float(align_angle)

def get_tomopitch_angle_offset():
    # Open tomopitch.log and get the suggested offset angle
    with open('tomopitch.log', 'r') as f:
        for line in f:
            if line.startswith(' Angle offset - Original:'):
                tomopitch_angle = line.split()[-1:][0]
    return float(tomopitch_angle)

def get_full_image():
    # Open up tilt.com file and find the line starting with FullImage
    with open('tilt.com', 'r') as f:
        for line in f:
            if line.startswith('FULLIMAGE'):
                full_image = ' '.join(line.split()[-2:])
    
    return full_image

def check_ccderaser(tomo_name):
    import subprocess
    # Run header on ${tomo_name}_ali.mrc and check if the output contains CCDERASER
    header = subprocess.check_output(['header', f'{tomo_name}_ali.mrc'])
    if 'CCDERASER' in header.decode('utf-8'):
        return True
    else:
        return False
    
def read_transforms_from_tomo_to_tilt(tlt_file, xf_file):
    import pandas as pd
    import numpy as np

    tilt_angles = pd.read_csv(tlt_file, sep="\s+", header=None)[0].to_numpy()

    transforms = pd.read_csv(xf_file, sep="\s+", header=None).to_numpy()
    transforms = transforms.reshape(-1, 2, 3, order="F")
    transforms = np.append(transforms, np.tile(np.array([0,0,1]),transforms.shape[0]).reshape(transforms.shape[0],1,3), axis=1)
    #print(transforms[0])
    #transforms = np.linalg.inv(transforms)
    #print(transforms[0])
    rotation_angles = np.rad2deg(np.arctan2(transforms[:,0,1], transforms[:,0,0]))
    #print(np.rad2deg(rotation_angles))
    translations = transforms[:,[0,1],2]
    return tilt_angles, rotation_angles, translations