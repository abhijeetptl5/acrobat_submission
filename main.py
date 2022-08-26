from utils import filtered_patches
from utils import base_transforms
from utils import foreground_detection_model
from utils import get_foreground
from args import infer_options
from utils import get_bbox
from utils import he_conv_input
from utils import ihc_conv_input
from utils import register
from utils import local_correction_samples
from utils import get_local_corrections
from utils import remove_smooth
from utils import remove_outliers
from utils import interpolate
import numpy as np

args = infer_options()
model = foreground_detection_model(args.foreground_model_path)

foreground = get_foreground(model, args.he_path, 
        batch_size=args.batch_size, 
        white_thr=args.white_thr, 
        black_thr=args.black_thr, 
        downsample=args.downsample,
        workers=args.workers)

_, bbox = get_bbox(foreground)
he_conv = he_conv_input(args.he_path, bbox, foreground, args.downsample)
ihc_conv = ihc_conv_input(args.ihc_path)
data = register(he_conv, ihc_conv)
df = local_correction_samples(foreground, bbox, data)
df = get_local_corrections(args.he_path, args.ihc_path, df)
df = remove_smooth(df)
df = remove_outliers(df)
interpolate(foreground, df, args.he_path)
