from mpi4py import MPI
import h5py
import numpy as np
from imseg.segment import ImSeg, kwarg
from imseg.threshold import otsu
from imseg.imtools.imclip import imclip_3D_f32, imclip_3D_f64


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# Read in arguments
if rank == 0:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path2data")
    parser.add_argument("--path2out")
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--otsu_bins", type=int, default=256)
    parser.add_argument("--clip_high", type=float, default=0.95)
    parser.add_argument("--clip_low", type=float, default=0.05)
    parser.add_argument("--input_type", type=int, default=1, help="Type of input data files. 1=binary, 2=hdf5.")
    args = parser.parse_args()
else:
    args = None
comm.bcast(args, root=0)

# Read data files
f = h5py.File(args.path2data, 'r')
dset = f['object']
dshape = dset.shape
num_proj = dshape[0]
dsize = dset.size
dtype = dset.dtype

# Calculate max & min of data
local_max = None
local_min = None
for i in xrange(rank, dshape[0], size):
    local_max = max(np.max(dset[i]), local_max)
    local_min = min(np.min(dset[i]), local_min)

# chunk_size = int(round(num_proj / size))
# if rank != size - 1:
#     local_max = np.max(dset[rank * chunk_size: (rank+1) * chunk_size])
#     local_min = np.min(dset[rank * chunk_size: (rank+1) * chunk_size])
# else:
#     local_max = np.max(dset[rank * chunk_size:])
#     local_min = np.min(dset[rank * chunk_size:])

global_max = None
global_min = None
comm.allreduce(local_max, global_max, op=MPI.MAX)
comm.allreduce(local_min, global_min, op=MPI.MAX)

# # Otsu threshold
# Calculate histogram
nbins = args.otsu_bins
local_hist = np.zeros(nbins)
for proj in xrange(rank, num_proj, size):
    local_hist += np.histogram(dset[proj], bins=nbins, range=(global_min, global_max))[0]
if rank == 0:
    hist = np.empty(nbins)
else:
    hist = None
comm.reduce(local_hist, hist, op=MPI.SUM, root=0)

# Calculate high and low values after clipping
bin_centers = otsu.bin_centers(np.linspace(global_min, global_max, nbins+1))
cumsum = np.cumsum(hist)
low_index = np.abs(cumsum - args.clip_low * dsize).argmin()
high_index = np.abs(cumsum - args.clip_high * dsize).argmin()
low_value, high_value = bin_centers[(low_index, high_index)]

# Calculate Otsu threshold
if rank == 0:
    # Clip the histogram
    hist[low_index] += cumsum[low_index - 1]
    hist[high_index] += cumsum[-1] - cumsum[high_index]
    hist[:low_index] = 0
    hist[high_index+1:] = 0
    # Calculate threshold
    threshold = otsu.threshold(hist, bin_centers)
else:
    threshold = None
comm.bcast(threshold, root=0)
if args.verbose:
    if rank == 0:
        print("Threshold = %s" % threshold)

# # Iterate through projections
# Initialize place holder
data_processed = np.empty(*dshape[1:], dtype=dtype)
for proj in xrange(rank, num_proj, size):
    # Clip & normalize data
    if dtype == np.float64:
        imclip_3D_f64.clip_norm(dset[proj], data_processed, low_value, high_value)
    elif dtype == np.float32:
        imclip_3D_f32.clip_norm(dset[proj], data_processed, low_value, high_value)
    # Save preprocessed data
    if args.verbose:
        print("Saving preprocessed data %s" % args.path2out+'/object_pre_'+str(rank))
    np.save(args.path2out+'/preprocessed/object_proj_'+str(proj), data_processed)

    # Segmentation
    if args.verbose:
        print("Segmenting projection %d on process %d ..." % (proj, rank))
    out_path = args.path2out + '/proj_%d/' % proj
    im_seg = ImSeg(data_processed)
    im_seg.init(thresholds=threshold,
                diff_sdf_kwargs=kwarg(it=5, dt=0.1),
                diff_error_kwargs=kwarg(it=20, dt=0.25),
                diff_ave_kwargs=kwarg(it=5, dt=0.1),
                reinit_sdf_kwargs=kwarg(niter=5, dt=0.1),
                init_reinit_sdf_kwargs=kwarg(niter=50, dt=0.1),
                im_slice=(slice(None), slice(None), 512 - 350),
                out_path=out_path, debug=True, verbose=args.verbose)
    im_seg.initialize()

    for i in xrange(10):
        im_seg.iterate(10)
        im_seg.save()

comm.Barier()
if rank == 0:
    print("All done!")


