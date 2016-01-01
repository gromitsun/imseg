import sys
import numpy as np
from imseg.segment import read_input, ImSeg

try:
    settings_file = sys.argv[1]
except IndexError:
    print('Error: expect input settings file!')
    sys.exit(-1)

settings = read_input(settings_file)
paras = read_input(settings['paras_file'])

im = np.fromfile(settings['path2data'])

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

while seg.iter_count < niter:
    seg.iterate(niter=1)
    if (seg.iter_count % niter_out) == 0:
        seg.save(prefix=settings['outdir'] + 'sdf_iter_', fmt='bin')
    if (seg.iter_count % niter_plot) == 0:
        seg.plot(outdir=settings['outdir'])
print('All done!')