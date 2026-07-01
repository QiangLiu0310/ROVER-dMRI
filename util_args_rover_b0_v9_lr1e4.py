import argparse
def get_args(cmd=True):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description="NODF Estimation for ROVER B0 data."
    )

    parser.add_argument(
        "--inr",
        action="store",
        type=str,
        help="Type of the INR - wire or siren or relu",
        choices=["siren", "wire", "relu"],
        default="siren",
    )

    parser.add_argument(
        "--device",
        action="store",
        type=str,
        help="Device.",
        default="cuda",
    )

    parser.add_argument(
        "--sh_order",
        action="store",
        default=8,
        type=int,
        help="Order of spherical harmonic basis",
    )

    parser.add_argument(
        "--bmarg",
        action="store",
        default=20,
        type=int,
        help="+= bmarg considered same b-value.",
    )

    parser.add_argument(
        "--rho",
        action="store",
        default=0.5,
        type=float,
        help="Length-scale parameter for Matern Prior.",
    )

    parser.add_argument(
        "--nu",
        action="store",
        default=1.5,
        type=float,
        help="Smoothness parameter for Matern Prior.",
    )

    parser.add_argument(
        "--num_epochs",
        action="store",
        default=8000, 
        type=int,
        help="Number of trainging epochs.",
    )

    parser.add_argument(
        "--learning_rate",
        action="store",
        default=0.0001,
        type=float,
        help="Learning rate for optimizer.",
    )

    parser.add_argument(
        "--calib_prop",
        action="store",
        default=0.1,
        type=float,
        help="Proportion of voxels to be used in posterior calibration.",
    )

    parser.add_argument(
        "--sigma2_mu",
        help="Variance for isotropic harmonic.",
        type=float,
        default=0.005,
    )

    parser.add_argument(
        "--sigma2_w", help="Variance parameter for GP prior.", type=float, default=0.5
    )

    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--deconvolve", action="store_true")

    parser.add_argument("--enable_schedulers", action="store_true")

    parser.add_argument("--simulation", action="store_true")

    parser.add_argument(
        "--omega0", help="SIREN frequency (higher=sharper).", type=float, default=30.0
    )

    parser.add_argument(
        "--omega0_hidden",
        help="SIREN frequency for hidden layers.",
        type=float,
        default=30.0,
    )

    parser.add_argument(
        "--sigma0", help="WIRE Sine frequency parameter.", type=float, default=5.0
    )

    parser.add_argument("--skip_conn", action="store_true")

    parser.add_argument("--batchnorm", action="store_true")

    parser.add_argument(
        "--num_workers",
        action="store",
        default=12,
        type=int,
        help="Number of workers for the dataloader.",
    )

    parser.add_argument(
        "--per_level_scale",
        action="store",
        default=1.5,
        type=float,
        help="Per level scale of resolution.",
    )

    parser.add_argument("--weight_decay", help="Weight decay.", type=float, default=0.0)

    parser.add_argument(
        "--view_num",
        default=12,
        type=int,
        help="number of view.",
    )

    parser.add_argument(
        "--img_size",
        default=[224, 224, 15],
        help="size of matrix for ROVER B0 data.",
    )

    parser.add_argument(
        "--img_path",
        action="store",
        default="/scratch/home/ql087/data_bwh/Philips/For_Qiang_phantom/nii_new/preprocess_rover_nii/imgs_nii_*.npy",
        type=str,
        help="NPY file path for ROVER B0 image data (nx X ny X nz)",
    )

    parser.add_argument(
        "--batch_size",
        action="store",
        default=100000,
        type=int,
        help="batch_size.",
    )

    parser.add_argument(
        "--image_save_iter",
        action="store",
        default=2000,
        type=int,
        help="image_save_iter.",
    )

    parser.add_argument(
        '--iter',
        type=int,
        default=15000,
        help="load model weights from iter",
    )

    # *** PSF depth-layer weighting mode ***
    parser.add_argument(
        "--weight_mode",
        action="store",
        type=str,
        default="equal",          # change to "equal" here to flip the default
        choices=["gaussian", "equal"],
        help=(
            "Depth-layer PSF weighting mode.\n"
            "  'gaussian' : Gaussian kernel (sigma=M/4), sum=1  [default]\n"
            "  'equal'    : uniform weights (1/M each),  sum=1"
        ),
    )

    if cmd:
        args = parser.parse_args()
    else:
        args, unknown = parser.parse_known_args()
    
    return args
