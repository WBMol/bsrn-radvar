import os

fdir_out = './output/'
fdir_images = './images/'

for fdir in [fdir_out, fdir_images]:
    os.makedirs(fdir) if not os.path.exists(fdir) else None
