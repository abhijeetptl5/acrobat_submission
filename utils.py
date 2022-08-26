import pandas as pd
from PIL import ImageOps
from PIL import ImageFilter
import cv2
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torchvision import models
import torch
from openslide import OpenSlide
from PIL import Image
from lr_utils import WSIDataset
import random
from scipy.interpolate import Rbf
from pathlib import Path
    
def base_transforms():
    transforms_ =  transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
    ])
    return transforms_


def filtered_patches(wsi, stride, white_thr, black_thr):
    h, w = wsi.dimensions
    thumb = np.array(wsi.get_thumbnail((h//stride, w//stride)).convert('L'))
    arr = np.logical_and(thumb < white_thr, thumb > black_thr)
    df = pd.DataFrame(columns=['dim1', 'dim2'])
    df['dim1'], df['dim2'] = stride*np.where(arr)[1]+(stride//2), stride*np.where(arr)[0]+(stride//2)
    return df


def foreground_detection_model(foreground_model_path):
    model = models.resnet18().cuda()
    model.fc = torch.nn.Linear(512, 2).cuda()
    model.load_state_dict(torch.load(foreground_model_path))
    model.eval()
    return model


def get_foreground(model, wsi_path, batch_size=512, white_thr=230, black_thr=20, stride=64, downsample=32, workers=8):
    wsi = OpenSlide(wsi_path)
    df = filtered_patches(wsi, stride, white_thr, black_thr)
    print(f'batches = {1 + (len(df)//batch_size)}')
    dataset = WSIDataset(df, wsi, base_transforms())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    preds = np.zeros(len(dataset))
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader)):
            out_ = model(data.cuda())
            preds[batch_size*i:batch_size*i+data.shape[0]] = torch.argmax(out_, axis=1).cpu().numpy()
    df['pred'] = preds.astype(int)
    w, h = wsi.dimensions
    tn = wsi.get_thumbnail((w//downsample, h//downsample)).convert('L')
    foreground = np.zeros_like(tn)
    df = df.loc[df['pred']==1].copy()
    df['dim1'] = df['dim1']//downsample
    df['dim2'] = df['dim2']//downsample
    foreground[df.values.T[:2].astype(int)[1], df.values.T[:2].astype(int)[0]] = 1
    return foreground


def bbox_helper(density, bins):
    dmap = []
    for i in range(0, len(density), len(density)//bins):
        dmap.append(np.sum(density[i: i+len(density)//bins]))
    for item in range(dmap.index(sorted(dmap)[-5]), len(dmap)):
        if dmap[item] < sorted(dmap)[-5]//20: break
    end = item+1 if item < len(density)-1 else item
    for item in range(dmap.index(sorted(dmap)[-5]), 0, -1):
        if dmap[item] < sorted(dmap)[-5]//20: break
    start = item-1 if item > 1 else item
    return start, end


def get_bbox_primary(foreground):
    density_x = np.sum(foreground, axis=0)
    bins = 100
    start_x, end_x = bbox_helper(density_x, bins)
    r, l = (len(density_x)//bins)*start_x, (len(density_x)//bins)*end_x
    foreground = foreground[:, r:l]
    density_y = np.sum(foreground, axis=1)
    start_y, end_y = bbox_helper(density_y, bins)
    t, b = (len(density_y)//bins)*start_y, (len(density_y)//bins)*end_y
    return foreground[t:b, :], (t, b, r, l)


def get_bbox_secondory(arr):
    r = np.where((np.cumsum(np.sum(arr, axis=0))/np.sum(arr))>0.05)[0][0]
    l = np.where((np.cumsum(np.sum(arr, axis=0))/np.sum(arr))<0.95)[-1][-1]

    t = np.where((np.cumsum(np.sum(arr, axis=1))/np.sum(arr))>0.05)[0][0]
    b = np.where((np.cumsum(np.sum(arr, axis=1))/np.sum(arr))<0.95)[-1][-1]

    return arr[t:b, r:l], (t, b, r, l)

def get_bbox(arr):
    bbox = get_bbox_primary(arr)
    if np.sum(bbox[0])/np.sum(arr)>0.8:
        return bbox
    bbox_sec = get_bbox_secondory(arr)
    if np.sum(bbox[0])/np.sum(arr) > np.sum(bbox_sec[0])/np.sum(arr):
        return bbox
    return bbox_sec


def he_conv_input(wsi, box, fg, stride):
    k = np.ones((7, 7))
    fg = cv2.dilate(fg, k, 1)
    bg = fg==0
    wsi = OpenSlide(wsi)
    w, h = wsi.dimensions
    tn = wsi.get_thumbnail((w//stride, h//stride)).convert('L')
    a, b, c, d = box
    bg = bg[a:b, c:d]
    image = Image.fromarray(np.array(tn)[a:b, c:d])
    mask = np.array(ImageOps.equalize(image, mask=None))<50
    image = ImageOps.invert(image)
    image = np.array(image)
    image[mask] = 255
    min_, max_ = np.min(image), np.max(image)
    image = (image-min_) / (max_-min_)
    image = (image + 1) / 2
    image = 255*image
    image[bg] = 0
    image = image.astype('uint8')
    return image

def ihc_conv_input(ihc_path):
    wsi = OpenSlide(ihc_path)
    w, h = wsi.dimensions
    tn = wsi.get_thumbnail((w//32, h//32)).convert('L')
    tn = np.array(tn)
    tn[tn<30] = 255
    tn = Image.fromarray(tn)
    tn = tn.filter(ImageFilter.BLUR)
    tn = np.array(tn)
    ihc = 255.0 - tn
    ihc = ihc.astype(float)
    return ihc


def pad_he(he_template):
    if he_template.shape[0]%2==1:
        pad = 1
        he_template = cv2.copyMakeBorder((he_template).astype('uint8'), pad, 0, 0, 0, 0)
    if he_template.shape[1]%2==1:
        pad = 1
        he_template = cv2.copyMakeBorder((he_template).astype('uint8'), 0, 0, 0, pad, 0)

    if he_template.shape[0]>he_template.shape[1]:
        pad = (he_template.shape[0]-he_template.shape[1])//2
        he_template = cv2.copyMakeBorder((he_template).astype('uint8'), 0, 0, pad, pad, 0)

    if he_template.shape[0]<he_template.shape[1]:
        pad = (he_template.shape[1]-he_template.shape[0])//2
        he_template = cv2.copyMakeBorder((he_template).astype('uint8'), pad, pad, 0, 0, 0)
        
    pad = max(list(he_template.shape)) // 4
    he_template = cv2.copyMakeBorder((he_template).astype('uint8'), pad, pad, pad, pad, 0)
    return he_template

def pad_ihc(ihc, he_template):
    pad = max(he_template.shape) // 2
    transform_ = transforms.Pad([pad, pad, pad, pad])
    ihc = Image.fromarray(ihc.astype('uint8'))
    ihc = transform_(ihc)
    ihc = np.array(ihc)
    ihc = torch.Tensor(ihc).cuda()
    return ihc

def rotation_matrix(he_template, astride, start, end):
    c = he_template.shape[1]//2, he_template.shape[0]//2
    num_planes = (end-start)//astride
    rot_matrix = torch.zeros((num_planes, he_template.shape[0], he_template.shape[1]), requires_grad=False).cuda()
    he_template = he_template.astype('uint8')
    for plane in range(num_planes):
        theta = start + (plane*astride)
        M = cv2.getRotationMatrix2D(c, theta, 1.0)
        rotated = cv2.warpAffine(he_template, M, (he_template.shape[1], he_template.shape[0]))
        rot_matrix[plane] = torch.Tensor(rotated.astype(float))
        torch.save(rot_matrix, 'rot_matrix_new.pt')
    return rot_matrix

def max_conv(ihc, rot_matrix, stride):
    max_ = -9999999
    reg_data = None
    for angle in range(rot_matrix.shape[0]):
        input_ = ihc[(None,)*2]
        weight = rot_matrix[angle][(None,)*2]
        out_ = torch.nn.functional.conv2d(input_, weight, stride=stride)
        if int(torch.max(out_))>max_:
            max_ = torch.max(out_)
            argmax = torch.where(out_==max_)
            reg_data = stride*int(argmax[2][0]), stride*int(argmax[3][0]), angle
    return reg_data

def register(he_template, ihc):
    he_template = pad_he(he_template)
    astride = 10
    stride = 10
    rot_matrix = rotation_matrix(he_template, astride, 0, 360)
    ihc = pad_ihc(ihc, he_template)
    x_strided, y_strided, angle_strided = max_conv(ihc, rot_matrix, stride)
    angle_strided = astride*angle_strided

    astride = 1
    stride = 1
    rot_matrix = rotation_matrix(he_template, astride, angle_strided-10, angle_strided+10)
    pad = max(he_template.shape) // 2
    ihc = ihc[x_strided-50:x_strided+50+(2*pad), y_strided-50:y_strided+50+(2*pad)]
    x_, y_, angle_ = max_conv(ihc, rot_matrix, stride)
    x_ = x_strided + x_ - 50
    y_ = y_strided + y_ - 50
    theta = angle_strided+angle_-10
    return x_, y_, theta


def local_correction_samples(foreground, bbox, reg_data):
    fg = foreground
    k = np.ones((7, 7))
    fg_dilate = cv2.dilate(fg, k, 1)
    fg_erode = cv2.erode(fg_dilate, k, 2)
    diff = fg_dilate - fg_erode
    diff = diff>0
    
    h, w = diff.shape

    sampled_points = []
    for i in range(0, w-32, 32):
        for j in range(0, h-32, 32):
            empty = np.zeros_like(diff)
            start = i, j
            end = i+32, j+32
            rect = cv2.rectangle(empty.astype('uint8'), start, end, 255, -1)
            rect = rect>0
            points = np.logical_and(rect, diff)
            num_points = len(np.where(points)[0])
            if num_points>0:
                sample = random.randint(0, num_points-1)
                x, y = np.where(points)[0][sample], np.where(points)[1][sample]
                sampled_points.append([x, y, 1])
            else:
                points = np.logical_and(rect, fg)
                num_points = len(np.where(points)[0])
                if num_points>0:
                    sample = random.randint(0, num_points-1)
                    x, y = np.where(points)[0][sample], np.where(points)[1][sample]
                    sampled_points.append([x, y, 0])
                    
    print(f"Number of sampled points = {len(sampled_points)}")

    sampled_points = np.array(sampled_points)
    sampled_points = 32 * sampled_points
    df = pd.DataFrame(columns=['hx', 'hy'])
    df['he_y'] = np.array(sampled_points).T[0]
    df['he_x'] = np.array(sampled_points).T[1]
    df['type'] = np.array(sampled_points).T[2]//32
    
    x_map = ((int(bbox[2]) + int(bbox[3])) // 2) - int(reg_data[1])
    y_map = ((int(bbox[0]) + int(bbox[1])) // 2) - int(reg_data[0])
    angle_map = -int(reg_data[2])
    cx_map = int(reg_data[1])
    cy_map = int(reg_data[0])

    df['x_map'] = x_map
    df['y_map'] = y_map
    df['angle_map'] = angle_map
    df['cx_map'] = cx_map
    df['cy_map'] = cy_map
    df['ihc_x'] = df['he_x'] - (32*df['x_map'])
    df['ihc_y'] = df['he_y'] - (32*df['y_map'])
    df['cx_map_'] = 32*df['cx_map']
    df['cy_map_'] = 32*df['cy_map']
    df['ihc_x_'] = df.apply(lambda row: get_rotation(row['ihc_x'], row['ihc_y'], row['angle_map'], row['cx_map_'], row['cy_map_'])[0], axis=1)
    df['ihc_y_'] = df.apply(lambda row: get_rotation(row['ihc_x'], row['ihc_y'], row['angle_map'], row['cx_map_'], row['cy_map_'])[1], axis=1)
    df['ihc_x'] = df['ihc_x_'].astype(int)
    df['ihc_y'] = df['ihc_y_'].astype(int)
    vals = (df[['he_x', 'he_y', 'ihc_x', 'ihc_y']].values).astype(int)
    df = df[['he_x', 'he_y', 'ihc_x', 'ihc_y', 'type']]
    df['theta'] = reg_data[2]
    # df = df.sample(frac=0.2)
    # df.reset_index(inplace=True)
    return df


def get_rotation(x, y, angle, cx, cy):
    x, y, angle, cx, cy = int(x) ,int(y), int(angle), int(cx), int(cy)
    M = cv2.getRotationMatrix2D((0, 0), angle, 1.0)[:, :2]
    return (np.array([x-cx, y-cy])@M)+np.array([cx, cy])


def he_patch(he_path, idx, hx, hy):
    he = OpenSlide(he_path)
    ps = 2048*2
    offset = ps // 2
    rs = ps // 16
    x, y = hx[idx], hy[idx]
    hp = he.read_region((x-offset, y-offset), 0, (ps, ps)).convert('L').resize((rs, rs))
    hp = np.array(hp)
    hp[hp<30] = 255
    hp[hp>230] = 255
    hp = 255-hp
    hp = Image.fromarray(hp)
    hp = transforms.ToTensor()(hp)
    hp_0 = hp.clone()
    hp_1 = hp.clone()
    hp_1[hp_1==0] = -1
    hp_0[hp_0==0] = 0
    hp_0 = hp_0[(None,)]
    hp_1 = hp_1[(None,)]
    return hp_0, hp_1


def ihc_patch(ihc_path, idx, ix, iy, theta):
    ihc = OpenSlide(ihc_path)
    ps = 2048*5
    offset = ps // 2
    rs = ps // 16
    x, y = ix[idx], iy[idx]
    ip = ihc.read_region((x-offset, y-offset), 0, (ps, ps)).convert('L').resize((rs, rs)).rotate(-theta)
    ip = np.array(ip)
    ip[ip<30] = 255
    ip[ip>230] = 255
    ip = 255-ip
    ip = Image.fromarray(ip)
    ip = transforms.ToTensor()(ip)
    ip_0 = ip.clone()
    ip_1 = ip.clone()
    ip_1[ip_1==0] = -1
    ip_0[ip_0==0] = 0
    ip_0 = ip_0[(None,)]
    ip_1 = ip_1[(None,)]
    return ip_0, ip_1


def get_local_corrections(he_path, ihc_path, df):
    df['xc_0'] = None
    df['yc_0'] = None
    df['xc_1'] = None
    df['yc_1'] = None
    df['h_0'] = None
    df['w_0'] = None
    df['h_1'] = None
    df['w_1'] = None
    df['area_0'] = None
    df['area_1'] = None

    hx, hy = df.values.T[0], df.values.T[1]
    ix, iy = df.values.T[2], df.values.T[3]
    theta = df['theta'][0]

    for idx in tqdm(range(len(df))):
        ip_0, ip_1 = ihc_patch(ihc_path, idx, ix, iy, theta)
        hp_0, hp_1 = he_patch(he_path, idx, hx, hy)
        out_0 = torch.nn.functional.conv2d(ip_0.cuda(), hp_0.cuda())
        out_1 = torch.nn.functional.conv2d(ip_1.cuda(), hp_1.cuda())
        np1 = out_1[0, 0].cpu().numpy()
        np0 = out_0[0, 0].cpu().numpy()
        max_0 = torch.where(out_0[0, 0]==torch.max(out_0))
        max_1 = torch.where(out_1[0, 0]==torch.max(out_1))
        xc_0, yc_0 = int(max_0[0][0])-192, int(max_0[1][0])-192
        xc_1, yc_1 = int(max_1[0][0])-192, int(max_1[1][0])-192
        bin_array = np0>(np.max(np0)*0.95)
        h_0 = np.max(np.where(bin_array)[0]) - np.min(np.where(bin_array)[0])
        w_0 = np.max(np.where(bin_array)[1]) - np.min(np.where(bin_array)[1])
        num_pixels_0 = np.sum(bin_array)
        m = 1.05 if np.max(np1)<0 else 0.95  
        bin_array = np1>(np.max(np1)*m)
        h_1 = np.max(np.where(bin_array)[0]) - np.min(np.where(bin_array)[0])
        w_1 = np.max(np.where(bin_array)[1]) - np.min(np.where(bin_array)[1])
        num_pixels_1 = np.sum(bin_array)
        df.iloc[idx, 6] = xc_0
        df.iloc[idx, 7] = yc_0
        df.iloc[idx, 8] = xc_1
        df.iloc[idx, 9] = yc_1
        df.iloc[idx, 10] = h_0
        df.iloc[idx, 11] = w_0
        df.iloc[idx, 12] = h_1
        df.iloc[idx, 13] = w_1
        df.iloc[idx, 14] = num_pixels_0
        df.iloc[idx, 15] = num_pixels_1
    return df


def remove_smooth(df):
    df.drop_duplicates(inplace=True)
    df['box_0'] = df['area_0']<500
    df['box_1'] = df['area_1']<500
    idx_thr = 160
    df['idx_0'] = (abs(df['xc_0'])<idx_thr)&(abs(df['yc_0'])<idx_thr)
    df['idx_1'] = (abs(df['xc_1'])<idx_thr)&(abs(df['yc_1'])<idx_thr)
    df['check_0'] = df['box_0']&df['idx_0']
    df['check_1'] = df['box_1']&df['idx_1']
    df = df.loc[df['check_0']|df['check_1']].copy()
    df['xc'] = df.apply(lambda row: select_xc(row['xc_0'], row['xc_1'], row['check_0'], row['check_1']), axis=1)
    df['yc'] = df.apply(lambda row: select_xc(row['yc_0'], row['yc_1'], row['check_0'], row['check_1']), axis=1)
    df = df.reset_index(drop=True)
    df = df[['he_x', 'he_y', 'xc', 'yc']]
    df['he_x'] //= 32
    df['he_y'] //= 32
    return df


def remove_outliers(df):
    hx = df.values.T[0].astype(int)
    hy = df.values.T[1].astype(int)
    xc = df.values.T[2].astype(int)
    yc = df.values.T[3].astype(int)

    potential_outliers = [100]

    while True:
        df['correction'] = np.sqrt(df['xc']**2 + df['yc']**2)
        mat = np.zeros((len(df), len(df)))
        for i in range(len(df)):
            ref = df.iloc[i]
            for j in range(len(df)):
                if i==j: continue
                compare = df.iloc[j]
                val_diff = np.sqrt((ref['xc']-compare['xc'])**2 + (ref['yc']-compare['yc'])**2)
                dist_diff = np.sqrt((ref['he_x']-compare['he_x'])**2 + (ref['he_y']-compare['he_y'])**2)
                slope = val_diff/dist_diff
                mat[i, j] = slope
        counts = np.unique(np.where(mat>0.3)[0], return_counts=True)
        potential_outliers = counts[1]>1
        if sum(potential_outliers)>0:
            indices = counts[0][potential_outliers]
            counts = counts[1][potential_outliers]
            outlier = indices[counts==max(counts)]
            outlier = outlier[0]
            df = df[~(df.index==outlier)]
            df = df.reset_index(drop=True)
            hx = df.values.T[0].astype(int)
            hy = df.values.T[1].astype(int)
            xc = df.values.T[2].astype(int)
            yc = df.values.T[3].astype(int)
        else: break
    return df

def select_xc(xc_0, xc_1, check_0, check_1):
    if check_1: return xc_1
    else: return xc_0



def interpolate(foreground, df, wsi_path):
    hx = df.values.T[0].astype(int)
    hy = df.values.T[1].astype(int)
    xc = df.values.T[2].astype(int)
    yc = df.values.T[3].astype(int)

    x_corrections = np.zeros_like(foreground)
    y_corrections = np.zeros_like(foreground)
    x_interpolated = np.zeros_like(foreground)
    y_interpolated = np.zeros_like(foreground)

    x_corrections[hy.astype(int), hx.astype(int)] = xc
    y_corrections[hy, hx] = yc

    rbfi_x = Rbf(hx, hy, xc, function='linear')
    rbfi_y = Rbf(hx, hy, yc, function='linear')
    for i in range(x_interpolated.shape[0]):
        x_interpolated[i] = rbfi_x(list(range(x_interpolated.shape[1])), [i]*x_interpolated.shape[1])
        y_interpolated[i] = rbfi_y(list(range(y_interpolated.shape[1])), [i]*y_interpolated.shape[1])
    np.save(f'{Path(wsi_path).stem}_x.npy', x_interpolated)
    np.save(f'{Path(wsi_path).stem}_y.npy', y_interpolated)
