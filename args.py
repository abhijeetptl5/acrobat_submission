import argparse

def infer_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--he_path', help='HE WSI Path', type=str, metavar='')
    parser.add_argument('--ihc_path', help='IHC WSI Path', type=str, metavar='')
    parser.add_argument('--batch_size', help='batch_size', type=int, default=512, metavar='')
    parser.add_argument('--workers', help='workers', type=int, default=8, metavar='')
    help_str = 'path for foreground detection model'
    model_path = 'foreground_detect.pt'
    parser.add_argument('--foreground_model_path', help=help_str, type=str, default=model_path, metavar='')
    parser.add_argument('--white_thr', help='Threshold for white pixel intensity', type=int, default=230, metavar='')
    parser.add_argument('--black_thr', help='Threshold for balck pixel intensity', type=int, default=20, metavar='')
    parser.add_argument('--stride', help='Stride for center-point selection', type=int, default=64, metavar='')
    parser.add_argument('--downsample', help='Downsamle factor', type=int, default=32, metavar='')

    return parser.parse_args()
