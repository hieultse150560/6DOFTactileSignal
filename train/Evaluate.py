import torch.nn as nn
import time
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import io, os
import argparse
from torch.utils.data import Dataset, DataLoader
from threeD_model_final import SpatialSoftmax3D, tile2openpose_conv3d
from threeD_dataLoader import sample_data_diffTask
from threeD_dataLoader import sample_data_diffTask_2
import pickle
import torch
import cv2
import h5py
from torch.optim.lr_scheduler import ReduceLROnPlateau
from progressbar import ProgressBar
from threeD_viz_video import generateVideo
from threeD_viz_image import generateImage
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--exp_dir', type=str, default='./', help='Experiment path') #Change
parser.add_argument('--exp', type=str, default='singlePeople', help='Name of experiment')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate') 
parser.add_argument('--batch_size', type=int, default=256, help='Batch size,256')
parser.add_argument('--weightdecay', type=float, default=1e-3, help='weight decay')
parser.add_argument('--window', type=int, default=10, help='window around the time step')
parser.add_argument('--subsample', type=int, default=1, help='subsample tile res')
parser.add_argument('--linkLoss', type=bool, default=True, help='use link loss') # Find min and max link
parser.add_argument('--epoch', type=int, default=500, help='The time steps you want to subsample the dataset to,500')
parser.add_argument('--ckpt', type=str, default ='singlePerson_0.0001_10_best', help='loaded ckpt file') # Enter link of trained model
parser.add_argument('--eval', type=bool, default=False, help='Set true if eval time') # Evaluation with test data. 2 Mode: Loading trained model and evaluate with test set, Training and Evaluation with evaluation set. 
parser.add_argument('--test_dir', type=str, default ='./', help='test data path') # Link to test data
parser.add_argument('--exp_image', type=bool, default=True, help='Set true if export predictions as images')
parser.add_argument('--exp_video', type=bool, default=True, help='Set true if export predictions as video')
parser.add_argument('--exp_data', type=bool, default=False, help='Set true if export predictions as raw data')
parser.add_argument('--exp_L2', type=bool, default=True, help='Set true if export L2 distance')
parser.add_argument('--train_continue', type=bool, default=False, help='Set true if eval time')
args = parser.parse_args()

# Vị trí thực khi chưa chuẩn hóa
def get_spatial_keypoint(keypoint):
    b = np.reshape(np.array([-100, -100, -1800]), (1,1,3))
    resolution = 100
    max = 19
    spatial_keypoint = keypoint * max * resolution + b
    return spatial_keypoint 

# Ước tính khoảng cách thực giữa dự đoán và kết quả
def get_keypoint_spatial_dis(keypoint_GT, keypoint_pred):
    dis = get_spatial_keypoint(keypoint_pred) - get_spatial_keypoint(keypoint_GT)
    # mean = np.reshape(np.mean(dis, axis=0), (21,3))
    return dis 

def remove_small(heatmap, threshold, device):
    z = torch.zeros(heatmap.shape[0], heatmap.shape[1], heatmap.shape[2], heatmap.shape[3], heatmap.shape[4]).to(device)
    heatmap = torch.where(heatmap<threshold, z, heatmap)
    return heatmap 

# use_gpu = torch.cuda.is_available()
# device = 'cuda:0' if use_gpu else 'cpu'
use_gpu = True
device = 'cuda:0'

np.random.seed(0)
torch.manual_seed(0)
model = tile2openpose_conv3d(args.window) # model
softmax = SpatialSoftmax3D(20, 20, 18, 21) # trả về heatmap và ước tính keypoint từ heatmap predicted

model.to(device)
softmax.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weightdecay)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.8, patience=5, verbose=True)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print (f"Total parameters: {pytorch_total_params}")
criterion = nn.MSELoss()

train_path = args.exp_dir + "batch_data/"
mask = []
test_dataset = sample_data_diffTask_2(train_path, args.window, args.subsample, "test")
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
print ("Test set size: ", len(test_dataset))
    
checkpoint = torch.load(args.exp_dir + 'ckpts/' + args.exp + '_' + str(args.lr)
                        + '_' + str(args.window) + '_best' + '.path.tar')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epochs = checkpoint['epoch']
loss = checkpoint['loss']
print("Loaded loss:", loss)
print("ckpt loaded:", args.ckpt)
print("Now running on test set")


model.eval()
avg_val_loss = []
avg_val_keypoint_l2_loss = []

tactile_GT = np.empty((1,96,96))
heatmap_GT = np.empty((1,21,20,20,18))
heatmap_pred = np.empty((1,21,20,20,18))
keypoint_GT = np.empty((1,21,3))
keypoint_pred = np.empty((1,21,3))
tactile_GT_v = np.empty((1,96,96))
heatmap_GT_v = np.empty((1,21,20,20,18))
heatmap_pred_v = np.empty((1,21,20,20,18))
keypoint_GT_v = np.empty((1,21,3))
keypoint_pred_v = np.empty((1,21,3))
keypoint_GT_log = np.empty((1,21,3))
keypoint_pred_log = np.empty((1,21,3))

bar = ProgressBar(max_value=len(test_dataloader)) # Thanh tiến trình

c = 0
for i_batch, sample_batched in bar(enumerate(test_dataloader, 0)):
    tactile = torch.tensor(sample_batched[0], dtype=torch.float, device=device)
    heatmap = torch.tensor(sample_batched[1], dtype=torch.float, device=device)
    keypoint = torch.tensor(sample_batched[2], dtype=torch.float, device=device)
    tactile_frame = torch.tensor(sample_batched[3], dtype=torch.float, device=device)


    with torch.set_grad_enabled(False):
        heatmap_out = model(tactile, device)
        heatmap_out = heatmap_out.reshape(-1, 21, 20, 20, 18) # Output shape từ model
        heatmap_transform = remove_small(heatmap_out.transpose(2,3), 1e-2, device)
        keypoint_out, heatmap_out2 = softmax(heatmap_transform) 

    loss_heatmap = torch.mean((heatmap_transform - heatmap)**2 * (heatmap + 0.5) * 2) * 1000 # Loss heatmap
    heatmap_out = heatmap_transform

    if i_batch % 100 == 0 and i_batch != 0:
        print (i_batch, loss_heatmap)
        # loss = loss_heatmap
        # print (loss)

    '''export image'''
    # Nếu có in ra hình ảnh kết quả để kiểm nghiệm
    if args.exp_image:
        base = 0
        imageData = [heatmap.cpu().data.numpy().reshape(-1,21,20,20,18),
                         heatmap_out.cpu().data.numpy().reshape(-1,21,20,20,18),
                         keypoint.cpu().data.numpy().reshape(-1,21,3),
                         keypoint_out.cpu().data.numpy().reshape(-1,21,3),
                         tactile_frame.cpu().data.numpy().reshape(-1,96,96)]

        generateImage(imageData, args.exp_dir + 'predictions/image/', i_batch, base)

    '''log data for L2 distance and video'''
    # Lưu lại chồng các frame để in ra video
    if args.exp_video:
        if i_batch>50 and i_batch<60: #set range
            heatmap_GT_v = np.append(heatmap_GT, heatmap.cpu().data.numpy().reshape(-1,21,20,20,18),axis=0)
            heatmap_pred_v = np.append(heatmap_pred, heatmap_out.cpu().data.numpy().reshape(-1,21,20,20,18),axis=0)
            keypoint_GT_v = np.append(keypoint_GT, keypoint.cpu().data.numpy().reshape(-1,21,3),axis=0)
            keypoint_pred_v = np.append(keypoint_pred, keypoint_out.cpu().data.numpy().reshape(-1,21,3),axis=0)
            tactile_GT_v = np.append(tactile_GT,tactile_frame.cpu().data.numpy().reshape(-1,96,96),axis=0)

    if args.exp_L2:
        keypoint_GT_log = np.append(keypoint_GT_log, keypoint.cpu().data.numpy().reshape(-1,21,3),axis=0)
        keypoint_pred_log = np.append(keypoint_pred_log, keypoint_out.cpu().data.numpy().reshape(-1,21,3),axis=0)

    '''save data'''
    # Nếu có lưu lại kết quả dự đoán để kiểm nghiệm (append)
    if args.exp_data:
        heatmap_GT = np.append(heatmap_GT, heatmap.cpu().data.numpy().reshape(-1,21,20,20,18),axis=0)
        heatmap_pred = np.append(heatmap_pred, heatmap_out.cpu().data.numpy().reshape(-1,21,20,20,18),axis=0)
        keypoint_GT = np.append(keypoint_GT, keypoint.cpu().data.numpy().reshape(-1,21,3),axis=0)
        keypoint_pred = np.append(keypoint_pred, keypoint_out.cpu().data.numpy().reshape(-1,21,3),axis=0)
        tactile_GT = np.append(tactile_GT,tactile_frame.cpu().data.numpy().reshape(-1,96,96),axis=0)

        if i_batch % 20 == 0 and i_batch != 0: #set the limit to avoid overflow
            c += 1
            toSave = [heatmap_GT[1:,:,:,:,:], heatmap_pred[1:,:,:,:,:],
                          keypoint_GT[1:,:,:], keypoint_pred[1:,:,:],
                          tactile_GT[1:,:,:]]
            pickle.dump(toSave, open(args.exp_dir + 'predictions/data/' + args.ckpt + str(c) + '.p', "wb"))
            tactile_GT = np.empty((1,96,96))
            heatmap_GT = np.empty((1,21,20,20,18))
            heatmap_pred = np.empty((1,21,20,20,18))
            keypoint_GT = np.empty((1,21,3))
            keypoint_pred = np.empty((1,21,3))

    avg_val_loss.append(loss.data.item())
print ("Loss:", np.mean(avg_val_loss))

# Nếu có lưu lại kết quả distance giữa các keypoint để kiểm nghiệm (sau khi đã xếp chồng)
if args.exp_L2:
    dis = get_keypoint_spatial_dis(keypoint_GT_log[1:,:,:], keypoint_pred_log[1:,:,:])
    pickle.dump(dis, open(args.exp_dir + 'predictions/L2/'+ args.ckpt + '_dis.p', "wb"))
    print ("keypoint_dis_saved:", dis, dis.shape)

# Tạo video
if args.exp_video:

    to_save = [heatmap_GT_v[1:,:,:,:,:], heatmap_pred_v[1:,:,:,:,:],
                  keypoint_GT_v[1:,:,:], keypoint_pred_v[1:,:,:],
                  tactile_GT_v[1:,:,:]]

    print (to_save[0].shape, to_save[1].shape, to_save[2].shape, to_save[3].shape, to_save[4].shape)

    generateVideo(to_save,
              args.exp_dir + 'predictions/video/' + args.ckpt,
              heatmap=True)
