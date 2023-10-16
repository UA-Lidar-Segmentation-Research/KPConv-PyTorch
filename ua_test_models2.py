#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on ModelNet40 dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import signal
import os
import numpy as np
import sys
import torch

# Dataset
from datasets.ModelNet40 import *
from datasets.S3DIS import *
from datasets.SemanticKitti import *
from datasets.Rellis import *
from torch.utils.data import DataLoader

from utils.config import Config
from utils.tester import ModelTester
from models.architectures import KPCNN, KPFCNN

from utils.ply import read_ply, write_ply

# Metrics
from utils.metrics import IoU_from_confusions, fast_confusion
from utils.config import Config
from sklearn.neighbors import KDTree

from models.blocks import KPConv


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

def model_choice(chosen_log):

    ###########################
    # Call the test initializer
    ###########################

    # Automatically retrieve the last trained model
    if chosen_log in ['last_ModelNet40', 'last_ShapeNetPart', 'last_S3DIS']:

        # Dataset name
        test_dataset = '_'.join(chosen_log.split('_')[1:])

        # List all training logs
        logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])

        # Find the last log of asked dataset
        for log in logs[::-1]:
            log_config = Config()
            log_config.load(log)
            if log_config.dataset.startswith(test_dataset):
                chosen_log = log
                break

        if chosen_log in ['last_ModelNet40', 'last_ShapeNetPart', 'last_S3DIS']:
            raise ValueError('No log of the dataset "' + test_dataset + '" found')

    # Check if log exists
    if not os.path.exists(chosen_log):
        raise ValueError('The given log does not exists: ' + chosen_log)

    return chosen_log

def map(label, mapdict=None):
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    # inv_map = {0: 0, 1: 3, 2: 4, 3: 5, 4: 6, 5: 8, 6: 15, 7: 17, 8: 18, 9: 19, 10: 23, 11: 27, 12: 31, 13: 33, 14: 34}
    # mapdict = inv_map

    if mapdict is None:
        mapdict = {0: 0, 1: 3, 2: 4, 3: 5, 4: 6, 5: 8, 6: 15, 7: 17, 8: 18, 9: 19, 10: 23, 11: 27, 12: 31, 13: 33,
                   14: 34}
    maxkey = 0
    for key, data in mapdict.items():
      if isinstance(data, list):
        nel = len(data)
      else:
        nel = 1
      if key > maxkey:
        maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
      lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
      lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
      try:
        lut[key] = data
      except IndexError:
        print("Wrong key ", key)
    # do the mapping
    return lut[label]

def slam_segmentation_test(net, test_loader, config, num_votes=1, debug=True):
    """
    Test method for slam segmentation models
    """

    device = torch.device("cuda:0")

    ############
    # Initialize
    ############

    # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
    test_smooth = 0.5
    last_min = -0.5
    softmax = torch.nn.Softmax(1)

    # Number of classes including ignored labels
    nc_tot = test_loader.dataset.num_classes
    nc_model = net.C

    # Test saving path
    test_path = None
    report_path = None
    if config.saving:
        test_path = join('test', config.saving_path.split('/')[-1])
        if not exists(test_path):
            makedirs(test_path)
        report_path = join(test_path, 'reports')
        if not exists(report_path):
            makedirs(report_path)

    if test_loader.dataset.set == 'validation':
        for folder in ['val_predictions', 'val_probs']:
            if not exists(join(test_path, folder)):
                makedirs(join(test_path, folder))
    else:
        for folder in ['predictions', 'probs']:
            if not exists(join(test_path, folder)):
                makedirs(join(test_path, folder))


    # Init validation container
    all_f_preds = []
    all_f_labels = []
    if test_loader.dataset.set == 'validation':
        for i, seq_frames in enumerate(test_loader.dataset.frames):
            all_f_preds.append([np.zeros((0,), dtype=np.int32) for _ in seq_frames])
            all_f_labels.append([np.zeros((0,), dtype=np.int32) for _ in seq_frames])

    #####################
    # Network predictions
    #####################

    predictions = []
    targets = []
    test_epoch = 0

    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(1)


    print('Initialize workers')
    for i, batch in enumerate(test_loader):
        # print(f"i: {i}, batch len: {batch.lengths}")

        # New time
        t = t[-1:]
        t += [time.time()]

        if i == 0:
            print('Done in {:.1f}s'.format(t[1] - t[0]))

        # if 'cuda' in device.type:
        batch.to(device)

        # Forward pass
        # Do the timing stuff here
        outputs = net(batch, config)

        # Get probs and labels
        stk_probs = softmax(outputs).cpu().detach().numpy()
        lengths = batch.lengths[0].cpu().numpy()
        f_inds = batch.frame_inds.cpu().numpy()
        r_inds_list = batch.reproj_inds
        r_mask_list = batch.reproj_masks
        labels_list = batch.val_labels
        torch.cuda.synchronize(device)

        t += [time.time()]

        # Get predictions and labels per instance
        # ***************************************

        i0 = 0
        # print(lengths)
        for b_i, length in enumerate(lengths):

            # Get prediction
            probs = stk_probs[i0:i0 + length]
            proj_inds = r_inds_list[b_i]
            proj_mask = r_mask_list[b_i]
            frame_labels = labels_list[b_i]
            s_ind = f_inds[b_i, 0]
            f_ind = f_inds[b_i, 1]

            # Project predictions on the frame points
            proj_probs = probs[proj_inds]

            # Safe check if only one point:
            if proj_probs.ndim < 2:
                proj_probs = np.expand_dims(proj_probs, 0)

            # Save probs in a binary file (uint8 format for lighter weight)
            seq_name = test_loader.dataset.sequences[s_ind]

            if not os.path.exists(os.path.join("/model/KPConv-PyTorch/output/sequences", seq_name, "predictions", "labels")):
                os.makedirs(os.path.join("/model/KPConv-PyTorch/output/sequences", seq_name, "predictions", "labels"))

            if test_loader.dataset.set == 'validation':
                folder = 'val_probs'
                pred_folder = 'val_predictions'
            else:
                folder = 'probs'
                pred_folder = 'predictions'
            filename = '{:s}_{:07d}.npy'.format(seq_name, f_ind)
            filepath = join(test_path, folder, filename)
            if exists(filepath):
                frame_probs_uint8 = np.load(filepath)
            else:
                frame_probs_uint8 = np.zeros((proj_mask.shape[0], nc_model), dtype=np.uint8)
            frame_probs = frame_probs_uint8[proj_mask, :].astype(np.float32) / 255
            frame_probs = test_smooth * frame_probs + (1 - test_smooth) * proj_probs
            frame_probs_uint8[proj_mask, :] = (frame_probs * 255).astype(np.uint8)
            np.save(filepath, frame_probs_uint8)

            # Save some prediction in ply format for visual
            if test_loader.dataset.set == 'validation':

                # Insert false columns for ignored labels
                frame_probs_uint8_bis = frame_probs_uint8.copy()
                for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                    if label_value in test_loader.dataset.ignored_labels:
                        frame_probs_uint8_bis = np.insert(frame_probs_uint8_bis, l_ind, 0, axis=1)

                # Predicted labels
                frame_preds = test_loader.dataset.label_values[np.argmax(frame_probs_uint8_bis,
                                                                         axis=1)].astype(np.int32)

                # Save some of the frame pots
                if True: # f_ind % 20 == 0:

                    # Insert false columns for ignored labels
                    for l_ind, label_value in enumerate(test_loader.dataset.label_values):
                        if label_value in test_loader.dataset.ignored_labels:
                            frame_probs_uint8 = np.insert(frame_probs_uint8, l_ind, 0, axis=1)

                    # Predicted labels
                    frame_preds = test_loader.dataset.label_values[np.argmax(frame_probs_uint8,
                                                                             axis=1)].astype(np.int32)

                    new_preds = map(frame_preds)

                    small_filename = '{:06d}.label'.format(f_ind)
                    path = join("/model/KPConv-PyTorch/output/sequences", seq_name, "predictions", "labels", small_filename)
                    new_preds.tofile(path)

                    seq_path = join(test_loader.dataset.path, 'sequences', test_loader.dataset.sequences[s_ind])
                    velo_file = join(seq_path, 'velodyne', test_loader.dataset.frames[s_ind][f_ind] + '.bin')
                    frame_points = np.fromfile(velo_file, dtype=np.float32)
                    frame_points = frame_points.reshape((-1, 4))
                    predpath = join(test_path, pred_folder, filename[:-4] + '.ply')

                # keep frame preds in memory
                all_f_preds[s_ind][f_ind] = frame_preds
                all_f_labels[s_ind][f_ind] = frame_labels

                # path = "test.label"
                # path = os.path.join(self.logdir, "sequences", batch
                #                     path_seq, "predictions", path_name)
                # frame_preds.tofile(path)
                # return

            else:
                print("Why am I here?")

            # Stack all prediction for this epoch
            i0 += length

        # Average timing
        t += [time.time()]
        mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

        # Display
        if (t[-1] - last_display) > 1.0:
            last_display = t[-1]
            message = 'e{:03d}-i{:04d} => {:.1f}% (timings : {:4.2f} {:4.2f} {:4.2f}) / pots {:d} => {:.1f}%'
            min_pot = int(torch.floor(torch.min(test_loader.dataset.potentials)))
            pot_num = torch.sum(test_loader.dataset.potentials > min_pot + 0.5).type(torch.int32).item()
            current_num = pot_num + (i + 1 - config.validation_size) * config.val_batch_num
            print(message.format(test_epoch, i,
                                 100 * i / config.validation_size,
                                 1000 * (mean_dt[0]),
                                 1000 * (mean_dt[1]),
                                 1000 * (mean_dt[2]),
                                 min_pot,
                                 100.0 * current_num / len(test_loader.dataset.potentials)))


    # Update minimum od potentials
    new_min = torch.min(test_loader.dataset.potentials)
    print('Test epoch {:d}, end. Min potential = {:.1f}'.format(test_epoch, new_min))

    #####################################
    # Results on the whole validation set
    #####################################

    # Confusions for our subparts of validation set
    Confs = np.zeros((len(predictions), nc_tot, nc_tot), dtype=np.int32)
    for i, (preds, truth) in enumerate(zip(predictions, targets)):

        # Confusions
        Confs[i, :, :] = fast_confusion(truth, preds, test_loader.dataset.label_values).astype(np.int32)


    # for sequence in all_f_labels:
    #     for frame in sequence:
    #         print(f"frame: {frame}, len: {len(frame)}")

    # Show vote results
    print('\nCompute confusion')

    val_preds = []
    val_labels = []
    t1 = time.time()
    # print(all_f_preds)
    # print(len(all_f_preds[0]))
    for i, seq_frames in enumerate(test_loader.dataset.frames):
        val_preds += [np.hstack(all_f_preds[i])]
        val_labels += [np.hstack(all_f_labels[i])]
        # print(f"seq_frames: {seq_frames}, len: {len(seq_frames)}")
        # print(f"label = {np.hstack(all_f_labels[i])}, len = {len(np.hstack(all_f_labels[i]))}, seq_frames = {seq_frames}, len = {len(seq_frames)}")
    val_preds = np.hstack(val_preds)
    val_labels = np.hstack(val_labels)
    t2 = time.time()
    C_tot = fast_confusion(val_labels, val_preds, test_loader.dataset.label_values)
    t3 = time.time()
    print(' Stacking time : {:.1f}s'.format(t2 - t1))
    print('Confusion time : {:.1f}s'.format(t3 - t2))

    s1 = '\n'
    for cc in C_tot:
        for c in cc:
            s1 += '{:7.0f} '.format(c)
        s1 += '\n'
    if debug:
        print(test_loader.dataset.label_values)
        print(test_loader.dataset.label_to_names)
        print(s1)

    # Remove ignored labels from confusions
    for l_ind, label_value in reversed(list(enumerate(test_loader.dataset.label_values))):
        if label_value in test_loader.dataset.ignored_labels:
            C_tot = np.delete(C_tot, l_ind, axis=0)
            C_tot = np.delete(C_tot, l_ind, axis=1)

    # Objects IoU
    val_IoUs = IoU_from_confusions(C_tot)

    # Compute IoUs
    mIoU = np.mean(val_IoUs)
    s2 = '{:5.2f} | '.format(100 * mIoU)
    for IoU in val_IoUs:
        s2 += '{:5.2f} '.format(100 * IoU)
    print(s2 + '\n')

    # Save a report
    report_file = join(report_path, 'report_{:04d}.txt'.format(int(np.floor(last_min))))
    str = 'Report of the confusion and metrics\n'
    str += '***********************************\n\n\n'
    str += 'Confusion matrix:\n\n'
    str += s1
    str += '\nIoU values:\n\n'
    str += s2
    str += '\n\n'
    with open(report_file, 'w') as f:
        f.write(str)

# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    ###############################
    # Choose the model to visualize
    ###############################

    chosen_log = sys.argv[1]

    # Choose the index of the checkpoint to load OR None if you want to load the current checkpoint
    chkp_idx = -1

    # Choose to test on validation or test split
    on_val = True

    # Deal with 'last_XXXXXX' choices
    chosen_log = model_choice(chosen_log)

    ############################
    # Initialize the environment
    ############################

    # Set which gpu is going to be used
    GPU_ID = '1'

    # Set GPU visible device
    # os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    ###############
    # Previous chkp
    ###############

    # Find all checkpoints in the chosen training folder
    chkp_path = os.path.join(chosen_log, 'checkpoints')
    chkps = [f for f in os.listdir(chkp_path) if f[:4] == 'chkp']

    # Find which snapshot to restore
    if chkp_idx is None:
        chosen_chkp = 'current_chkp.tar'
    else:
        chosen_chkp = np.sort(chkps)[chkp_idx]
    chosen_chkp = os.path.join(chosen_log, 'checkpoints', chosen_chkp)

    # Initialize configuration class
    config = Config()
    config.load(chosen_log)

    print(config.dataset)

    # inv_map = {0: 0, 1: 3, 2: 4, 3: 5, 4: 6, 5: 8, 6: 15, 7: 17, 8: 18, 9: 19, 10: 23, 11: 27, 12: 31, 13: 33, 14: 34}

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.

    #config.augment_noise = 0.0001
    config.augment_symmetries = False
    #config.batch_num = 3
    #config.in_radius = 4

    config.validation_size = 3343*4
    config.input_threads = 1
    config.num_classes = 14

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    if on_val:
        set = 'validation'
    else:
        set = 'test'

    # Initiate dataset
    if config.dataset == 'SemanticKitti':
        test_dataset = SemanticKittiDataset(config, set=set, balance_classes=False)
        test_sampler = SemanticKittiSampler(test_dataset)
        collate_fn = SemanticKittiCollate
    elif config.dataset == 'Rellis':
        print("RELLIS")
        test_dataset = RellisDataset(config, set=set, balance_classes=False)
        test_sampler = RellisSampler(test_dataset)
        # test_sampler = torch.utils.data.SequentialSampler(test_dataset)
        collate_fn = RellisCollate
    else:
        raise ValueError('Unsupported dataset : ' + config.dataset)

    print(f"Running on {set} set with {len(test_dataset)} samples.")
    # Data loader
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=collate_fn,
                             num_workers=1,
                             pin_memory=True)

    # Calibrate samplers
    test_sampler.calibration(test_loader, verbose=True)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)

    print('\nStart test')
    print('**********\n')
    device = torch.device("cuda:0")

    net.to(device)

    ##########################
    # Load previous checkpoint
    ##########################

    checkpoint = torch.load(chosen_chkp)
    net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    net.eval()
    print("Model and training state restored.")

    # print(test_loader.dataset[0])
    # print(len(test_loader.dataset[2]))
    # test, test2 = enumerate(test_loader)
    # print(test, test2)

    for i in test_loader:
        slam_segmentation_test(net, test_loader, config)