import logging
import sys

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import re
import sys
import cv2
import numpy as np
import random

def seed_everything(seed):
    torch.manual_seed(seed)       # Current CPU
    torch.cuda.manual_seed(seed)  # Current GPU
    np.random.seed(seed)          # Numpy module
    random.seed(seed)             # Python random module
    torch.backends.cudnn.benchmark = False    # Close optimization
    torch.backends.cudnn.deterministic = True # Close optimization
    torch.cuda.manual_seed_all(seed) # All GPU (Optional)


def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    # exlude invalid pixels and extremely large diplacements
    mag = torch.sum(flow_gt ** 2, dim=1, keepdim=True).sqrt()

    # exclude extremly large displacements
    valid = ((valid >= 0.5) & (mag < max_flow))
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        adjusted_loss_gamma = loss_gamma ** (15 / (n_predictions))
        i_weight = adjusted_loss_gamma ** (n_predictions - i)
        i_loss = (flow_preds[i] - flow_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
        flow_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((flow_preds[-1] - flow_gt) ** 2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def fetch_optimizer(args, model, last_epoch=-1, checkpoint=None):
    """ Create the optimizer and learning rate scheduler """
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.wdecay, eps=1e-8)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100, pct_start=0.01,
                                              cycle_momentum=False, anneal_strategy='linear', last_epoch=last_epoch)

    return optimizer, scheduler


class Logger:
    SUM_FREQ = 100

    def __init__(self, model, scheduler, name):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.log_dir = 'runs/' + name
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def _print_training_status(self):
        metrics_data = [self.running_loss[k] / Logger.SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps + 1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, " * len(metrics_data)).format(*metrics_data)

        # print the training status
        logging.info(f"Training Metrics ({self.total_steps}): {training_str + metrics_str}")

        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)

        for k in self.running_loss:
            self.writer.add_scalar("train/" + k, self.running_loss[k] / Logger.SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % Logger.SUM_FREQ == Logger.SUM_FREQ - 1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir)

        for key in results:
            self.writer.add_scalar("valid/" + key, results[key], self.total_steps)

    def close(self):
        self.writer.close()

def resizedImg(img, percent):
    width = int(img.shape[1] * percent / 100)
    height = int(img.shape[0] * percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized

def extractID(filename):
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    return None

def writeTxt(filePath, array):
    # Write the array to a text file with formatting
    np.savetxt(filePath, array, fmt='%.2f')
    # Confirm the content of the file
    with open(filePath, 'r') as file:
        content = file.read()

def loadStereoMaps(folderPath,stereoMapFile):
    stereoMaps = [] 
    cvFile = cv2.FileStorage(os.path.join(folderPath, stereoMapFile), cv2.FileStorage_READ)
    # print("cvFile: ",os.path.join(folderPath, stereoMapFile))
    stereoMapR_x = cvFile.getNode('stereoMapR_x').mat()
    stereoMapR_y = cvFile.getNode('stereoMapR_y').mat()
    stereoMapL_x = cvFile.getNode('stereoMapL_x').mat()
    stereoMapL_y = cvFile.getNode('stereoMapL_y').mat()
    Q = cvFile.getNode('q').mat()
    cvFile.release()
    stereoMaps = [stereoMapL_x, stereoMapL_y, stereoMapR_x, stereoMapR_y]
    return stereoMaps, Q

def mapImageLR(imgL, imgR, stereoMaps):
    correctedImgL = cv2.remap(imgL, stereoMaps[0], stereoMaps[1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    correctedImgR = cv2.remap(imgR, stereoMaps[2], stereoMaps[3], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return correctedImgL, correctedImgR

def findFarthestPoints(contour):
 
    # Find the convex hull of the contour
    hull = cv2.convexHull(contour, returnPoints=True)

    # Use the Rotating Calipers algorithm to find the farthest points
    max_distance = 0
    farthest_points = (None, None)

    for i in range(len(hull)):
        for j in range(i + 1, len(hull)):
            point1 = tuple(hull[i][0])
            point2 = tuple(hull[j][0])
            distance = np.linalg.norm(np.array(point1) - np.array(point2))

            if distance > max_distance:
                max_distance = distance
                farthest_points = (point1, point2)

    return farthest_points
def reCentroid(mask, points):
    h, w = mask.shape[:2]
    rePoints = []
    # print("points: ",points)
    for pi, p in enumerate(points):
        maskImg = mask.copy()
        blackImage = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(blackImage, p, 20, (255, 0, 0), -1)
        mask_ = cv2.bitwise_and(maskImg,blackImage)
        _, thresh = cv2.threshold(mask_, 127, 255, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mCnt = max(contours, key = cv2.contourArea)
        M = cv2.moments(mCnt)
        centroidX = int(M['m10'] / (M['m00']+1e-10))
        centroidY = int(M['m01'] / (M['m00']+1e-10))
        rePoints.append([centroidX, centroidY])
    # print("rePoints: ",rePoints)
    return rePoints
    # cv2.imshow("mask_",resizedImg(mask_,25))

def findMaxContour(image, objNum =2, iter=1):
    h, w = image.shape[:2]
    kernel = np.ones((3, 3), np.uint8)
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(cv2.erode(thresh, kernel, iterations=iter), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sortedContours = sorted(contours, key=cv2.contourArea, reverse=True)
    cnts = [sortedContours[0],sortedContours[1]]
    blackImage = np.zeros((h, w), dtype=np.uint8)
    fPoint1 = []
    fPoint2 = []
    if len(sortedContours) >= 2 and objNum ==2:
        fPoint1 = findFarthestPoints(sortedContours[0])
        fPoint2 = findFarthestPoints(sortedContours[1])
        cnts = [sortedContours[0],sortedContours[1]]
        

    else: 
        cnts = sortedContours
    # print(cnts)
    maskImg = cv2.drawContours(blackImage, cnts, -1, 255, cv2.FILLED)
    fPoint1_ = reCentroid(maskImg, fPoint1)
    fPoint2_ = reCentroid(maskImg, fPoint2)
    fPoints = [fPoint1_,fPoint2_]
    print("fPoints",[fPoint1,fPoint2])
    print("fPoints", fPoints)

    # cv2.circle(maskImg, farthestPoint1[0], 10, (255, 0, 0), -1)
    # cv2.circle(maskImg, farthestPoint1[1], 10, (255, 0, 0), -1)
    # cv2.circle(maskImg, farthestPoint2[0], 10, (255, 0, 0), -1)
    # cv2.circle(maskImg, farthestPoint2[1], 10, (255, 0, 0), -1)
    return fPoints, maskImg
def generate_dis(imgL, imgR, minDisp, numDisp):
    """
    Computes disparity using Semi-Global Block Matching (SGBM) algorithm.

    Returns:
    np.ndarray: Disparity map.
    """
    if imgL.ndim == 2:
        imgChannels = 1
    else:
        imgChannels = 3
    blockSize = 7 #7
    param = {'minDisparity': minDisp,
                'numDisparities': numDisp,
                'blockSize': blockSize,
                'P1': 8 * imgChannels * blockSize ** 2,
                'P2': 32 * imgChannels * blockSize ** 2,
                'disp12MaxDiff': 1, 
                'preFilterCap': 63, 
                'uniquenessRatio': 12, 
                'speckleWindowSize': 400, 
                'speckleRange': 2,
                'mode': cv2.STEREO_SGBM_MODE_SGBM_3WAY
                }
    
    sgbm = cv2.StereoSGBM_create(**param)
    # self._cropMask()
    imageL =  cv2.cvtColor(imgL,cv2.COLOR_BGR2GRAY)
    imageR = cv2.cvtColor(imgR,cv2.COLOR_BGR2GRAY)

    # if self.imgPoints is not None:
    #     imageL, imageR, bbox = self.roiCrop()
    disparity = sgbm.compute(imageL, imageR)
    
    disparity = disparity.astype(np.float32) / 16
    disparity = cv2.bilateralFilter(disparity, 19, 75,75)
    # disparity = disparity.astype(np.float32)
    # if self.imgPoints is not None:
    #     minDisparity = np.min(disparity)
    #     blankDisparity = np.zeros((self.h,self.w), dtype=np.uint8)*minDisparity
    #     blankDisparity[bbox[1]:bbox[1]+bbox[3],bbox[0]:bbox[0]+bbox[2]] = disparity
        
    #     return blankDisparity
    disp_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    disp_colored = cv2.applyColorMap(np.uint8(disp_normalized), cv2.COLORMAP_JET) 
    
    return disparity, disp_colored
def mask_left_image(imgL, disparity, min_disp=256, max_disp=512):
    """Keep only features in the left image where disparity is within a range."""
    # Normalize disparity for better visualization
    disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_norm = np.uint8(disp_norm)

    # Create a mask where disparity is within the desired range
    mask = (disparity >= min_disp) & (disparity <= max_disp)
    
    # Convert mask to 3 channels (for RGB images)
    mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    # Apply mask to the left image
    masked_img = imgL * mask_3ch  # Keeps only valid disparity regions

    return masked_img

def soft_remove_background(imgL, disparity, min_disp=10, max_disp=200, blur_size=15):
    """Remove background with a soft edge transition."""
    # Normalize disparity to [0, 255] for better processing
    # disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Create a weight mask where valid disparities are kept (1) and others fade out
    mask = np.zeros_like(disparity, dtype=np.float32)
    mask[(disparity >= min_disp) & (disparity <= max_disp)] = 1.0  # Valid pixels = 1

    # Apply Gaussian blur to create soft edges
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)

    # Convert mask to 3 channels for RGB blending
    mask_3ch = np.repeat(mask[:, :, np.newaxis], 3, axis=2)

    # Create a white background
    white_bg = np.ones_like(imgL, dtype=np.uint8) * 255

    # Blend the left image with the white background using the soft mask
    result = (imgL * mask_3ch + white_bg * (1 - mask_3ch)).astype(np.uint8)

    return result
def bluring_map(img, disparity, threshold = 100):
    # Normalize the disparity map (convert to depth-like values)
    depth_map = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Define depth threshold (adjust based on scene)
    depth_threshold = threshold  # Pixels with depth < threshold remain sharp

    # Create a mask for sharp regions
    sharp_mask = depth_map > depth_threshold  # True for regions that should be sharp

    # Apply Gaussian Blur to the entire image
    blurred_img = cv2.medianBlur(img, 45)

    # Merge sharp and blurred regions
    final_img = img.copy()
    final_img[~sharp_mask] = blurred_img[~sharp_mask]
    return final_img
