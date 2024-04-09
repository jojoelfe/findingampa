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
    with open('align.com', 'r') as f:
        for line in f:
            if line.startswith('AngleOffset'):
                align_angle = line.split()[-1:][0]
    
    return float(align_angle)

def get_tomopitch_angle_offset():
    # Open tomopitch.log and get the suggested offset angle
    with open('tomopitch.log', 'r') as f:
        for line in f:
            if line.startswith(' Angle offset - Original:'):
                tomopitch_angle = line.split()[-1:][0]
    return float(tomopitch_angle)