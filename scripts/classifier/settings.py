import os

fdir_img = './images/'
fdir_data_base = '/run/media/wouter/ResearchData/'
fdir_data_in_fmt = os.path.join(fdir_data_base, 'Cabauw', 'BSRN', '{res}', 'processed', '{y}', '{m:02d}')
fdir_data_msg_fmt = os.path.join(fdir_data_base, 'eumetsat', 'msgcpp', 'timeseries', '{y}', '{m:02d}')
fdir_data_out = './validation/'
fdir_data_out_debug = os.path.join(fdir_data_out, 'criteria')

MIN_VERSION_REQ = '0.8'

for fdir in [fdir_img, fdir_data_out, fdir_data_out_debug]:
    os.makedirs(fdir) if not os.path.exists(fdir) else None
