import sys
import os
import glob
import numpy as np
from time import time
import datetime
from imseg.segment import read_input, ImSeg, proc_slice, dump_dict, dump_file

try:
    settings_file = sys.argv[1]
except IndexError:
    print('Error: expect input settings file!')
    sys.exit(-1)

print('Starting program ...\n Time now is {}'.format(datetime.datetime.now()))

# # Read and process input files
print('Reading settings file %s ...' % settings_file)
settings = read_input(settings_file)
print('Reading parameters file %s ...' % settings['paras_file'])
paras = read_input(settings['paras_file'])

# Copy input files to output directory
dump_file(settings_file, settings['outdir']+'settings.txt')
dump_file(settings['paras_file'], settings['outdir']+'parameters.txt')

print('* * * Settings * * *')
print(dump_dict(settings))
print('* * * Parameters * * *')
print(dump_dict(paras))

data_slice = settings['data_slice']
for i, x in enumerate(data_slice):
    data_slice[i] = proc_slice(x)

if type(settings['path2data']) is not list:
    data_file_list = sorted(glob.glob(settings['path2data']))
else:
    data_file_list = []
    for i in settings['path2data']:
        data_file_list.append(sorted(glob.glob(i)))

for data_file in data_file_list:
    outsubdir = os.path.splitext(os.path.basename(data_file))[0]
    outdir = os.path.join(settings['outdir'], outsubdir)
    # Read data file
    print('Reading data file %s ...' % data_file)
    im = np.fromfile(data_file, dtype=settings['dtype']).reshape(settings['data_shape'])[data_slice]

    print('Initializing ImSeg object ...')
    seg = ImSeg(im, **paras)

    if 'continue' in settings:
        seg.load(settings['continue'],
                 iter_count=settings.get('iter_count'),
                 thresholds=settings.get('thresholds'))
    else:
        seg.initialize()

    niter_plot = settings['niter_plot']
    niter_out = settings['niter_out']
    niter = settings['niter']

    plot_slice = settings['plot_slice']
    for i, x in enumerate(plot_slice):
        plot_slice[i] = proc_slice(x)

    while seg.iter_count < niter:
        start = time()
        seg.iterate(niter=1)
        print('Completed 1 iteration in {} seconds.'.format(time()-start))
        if (seg.iter_count % niter_out) == 0:
            seg.save(prefix=outdir + 'sdf_iter_', fmt='bin')
        if (seg.iter_count % niter_plot) == 0:
            seg.plot(outdir=outdir, im_slice=plot_slice)

print('Program finished!')
print('Time now is {}'.format(datetime.datetime.now()))
