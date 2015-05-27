import os
import glob
from mpi4py import MPI
import numpy as np
from imseg.segment import ImSeg, kwarg
from imseg.threshold import otsu
from imseg.imtools.imclip import imclip_1D_f32, imclip_1D_f64


def path_to_num(path, prefix="object_", suffix=".bin"):
    return int(os.path.basename(path).strip(prefix).strip(suffix))


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
    parser.add_argument("--overwrite", type=bool, default=False)
    parser.add_argument("--otsu_bins", type=int, default=256)
    parser.add_argument("--clip_high", type=float, default=0.95)
    parser.add_argument("--clip_low", type=float, default=0.05)
    parser.add_argument("--input_type", type=int, default=1, help="Type of input data files. 1=binary, 2=hdf5.")
    parser.add_argument("--z_num", type=int, help="Number of pixels in z.")
    parser.add_argument("--y_num", type=int, help="Number of pixels in y.")
    parser.add_argument("--x_num", type=int, help="Number of pixels in x.")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterations used in segmentation.")
    parser.add_argument("--load", type=int, default=0,
                        help="Load previous finished segmentation files with iteration number specified.")
    args = parser.parse_args()
else:
    args = None
args = comm.bcast(args, root=0)

# Read data files
path2data = args.path2data
z_num = args.z_num
y_num = args.y_num
x_num = args.x_num
flist = glob.glob(path2data + "/*.bin")
t_num = len(flist)
dsize = t_num * z_num * y_num * x_num
dtype = np.float32

# Calculate max & min of data
local_max = - np.inf
local_min = np.inf
for i in xrange(rank, t_num, size):
    if args.verbose:
        print("Opening file %s" % flist[i])
    dset = np.fromfile(flist[i], dtype=dtype)
    local_max = max(np.max(dset), local_max)
    local_min = min(np.min(dset), local_min)
    if args.verbose:
        print "local min", local_min, "local max", local_max

# chunk_size = int(round(num_proj / size))
# if rank != size - 1:
#     local_max = np.max(dset[rank * chunk_size: (rank+1) * chunk_size])
#     local_min = np.min(dset[rank * chunk_size: (rank+1) * chunk_size])
# else:
#     local_max = np.max(dset[rank * chunk_size:])
#     local_min = np.min(dset[rank * chunk_size:])

global_max = None
global_min = None
global_max = comm.allreduce(local_max, global_max, op=MPI.MAX)
global_min = comm.allreduce(local_min, global_min, op=MPI.MAX)

# global_min = -0.0065692
# global_max = 0.00650802
if args.verbose:
    if rank == 0:
        print "global min", global_min, "global max", global_max
    comm.Barrier()

# # Otsu threshold
# Calculate histogram
if args.verbose:
    if rank == 0:
        print("Begin calculating histograms ...")
    comm.Barrier()
nbins = args.otsu_bins
local_hist = np.zeros(nbins, dtype=int)
# t_num = 2 ###
# dsize = t_num * z_num * y_num * x_num
# flist = flist[100:]
for i in xrange(rank, t_num, size):
    if args.verbose:
        print("Opening file %s" % flist[i])
    dset = np.fromfile(flist[i], dtype=dtype)
    local_hist += np.histogram(dset, bins=nbins, range=(global_min, global_max))[0]
if rank == 0:
    hist = np.empty(nbins, dtype=int)
else:
    hist = None
    # hist = np.empty(nbins, dtype=int)
comm.Reduce(local_hist, hist, op=MPI.SUM, root=0)

if rank == 0:
    # Calculate high and low values after clipping
    if args.verbose:
        print("Calculating low and high values after clipping ...")
    bin_centers = otsu.bin_centers(np.linspace(global_min, global_max, nbins+1))
    cumsum = np.cumsum(hist)
    low_index = np.abs(cumsum - (args.clip_low * dsize)).argmin()
    high_index = np.abs(cumsum - (args.clip_high * dsize)).argmin()
    print bin_centers.shape, low_index, high_index, args.clip_high, args.clip_low, dsize
    print cumsum
    print np.abs(cumsum - args.clip_low * dsize)
    print np.abs(cumsum - args.clip_high * dsize)
    low_value, high_value = bin_centers[[low_index, high_index]]
    if args.verbose:
        print("Clipped data will have low = %s and high = %s" % (low_value, high_value))

    # Calculate Otsu threshold
    if args.verbose:
        print("Begin calculating Ostu threshold ...")
    # Clip the histogram
    if args.verbose:
        print("Clipping histogram ...")
    hist[low_index] += cumsum[low_index - 1]
    hist[high_index] += cumsum[-1] - cumsum[high_index]
    hist[:low_index] = 0
    hist[high_index+1:] = 0
    # Calculate threshold
    if args.verbose:
        print("Calculating Ostu threshold ...")
    threshold = otsu.threshold(hist, bin_centers)
    if args.verbose:
        print("Threshold (before normalization) = %s" % threshold)
    threshold = (threshold - low_value) / (high_value - low_value)
    if args.verbose:
        print("Threshold (after normalization) = %s" % threshold)
else:
    threshold = None
    low_value = None
    high_value = None
threshold = comm.bcast(threshold, root=0)
low_value = comm.bcast(low_value, root=0)
high_value = comm.bcast(high_value, root=0)

# threshold = 0.428571428571
# low_value = -0.000209380117187
# high_value = 0.000148200117188
# threshold = 0.444444444444
# low_value = -0.000221107149628
# high_value = 0.000154846320584
# # Iterate through time frames
# Initialize place holder
data_processed = np.empty((z_num * y_num * x_num), dtype=dtype)
dir_preprocessed = args.path2out + "/preprocessed/"
for i in xrange(rank, t_num, size):
    # Input and ouput paths
    t_frame = path_to_num(flist[i])
    file_preprocessed = dir_preprocessed + "/object_pre_" + str(t_frame) + ".bin"
    dir_seg = args.path2out + '/tframe_%d/' % t_frame
    # Check if already finished
    file_npz = dir_seg+'/imseg_iter_%d.npz' % args.iterations
    if (not args.overwrite) and (os.path.exists(file_npz)):
        if args.verbose:
            print("Skipped tframe %d, iteration %d. File %s already exists." % (t_frame, args.iterations, file_npz))
        continue
    # Preprocess data
    if args.overwrite or (not os.path.exists(file_preprocessed)):
        if args.verbose:
            print("Opening file %s" % flist[i])
        dset = np.fromfile(flist[i], dtype=dtype)
        # Clip & normalize data
        if args.verbose:
            print("Clipping and normalizing data with low = %s and high = %s" % (low_value, high_value))
        if dtype == np.float64:
            imclip_1D_f64.clip_norm(dset, data_processed, low_value, high_value)
        elif dtype == np.float32:
            imclip_1D_f32.clip_norm(dset, data_processed, low_value, high_value)
        # Save preprocessed data
        if rank == 0:
            if not os.path.exists(dir_preprocessed):
                if args.verbose:
                    print("Creating directory: %s" % dir_preprocessed)
                os.makedirs(dir_preprocessed)
                comm.Barrier()
        if args.verbose:
            print("Saving preprocessed data %s" % file_preprocessed)
        fout = open(file_preprocessed, 'wb')
        fout.write(data_processed.tobytes())
        fout.close()
    else:
        if args.verbose:
            print("Reading preprocessed data from %s ..." % file_preprocessed)
        data_processed = np.fromfile(file_preprocessed, dtype=dtype)

    # Segmentation
    if args.verbose:
        print("Segmenting projection %d on process %d ..." % (t_frame, rank))
    im_seg = ImSeg(data_processed.reshape(z_num, y_num, x_num))
    im_seg.init(thresholds=threshold,
                diff_sdf_kwargs=kwarg(it=5, dt=0.1),
                diff_error_kwargs=kwarg(it=20, dt=0.25),
                diff_ave_kwargs=kwarg(it=5, dt=0.1),
                reinit_sdf_kwargs=kwarg(niter=5, dt=0.1),
                init_reinit_sdf_kwargs=kwarg(niter=50, dt=0.1),
                im_slice=(slice(None), slice(None), 512),
                out_path=dir_seg, debug=True, verbose=args.verbose)
    # Attempt to load earlier results
    file_load = dir_seg+'/imseg_iter_%d.npz' % args.load
    if args.load and os.path.exists(file_load):
        if args.verbose:
            print("Loading segmentation result from %s ..." % file_load)
        im_seg.load(file_load)
        iterations = args.iterations - args.load
    else:
        iterations = args.iterations
        im_seg.initialize()

    # for i in xrange(10):
    #     im_seg.iterate(10)
    #     im_seg.save()
    im_seg.iterate(iterations)
    im_seg.save()
comm.Barrier()
if rank == 0:
    print("All done!")
