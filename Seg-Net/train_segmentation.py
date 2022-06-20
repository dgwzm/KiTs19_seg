import numpy
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataio.loader import get_dataset, get_dataset_path
from dataio.transformation import get_dataset_transformation
from utils.util import json_file_to_pyobj
from utils.visualiser import Visualiser,ViShow
from utils.error_logger import ErrorLogger
import os
from models import get_model

def train(arguments):

    # Parse input arguments
    json_filename = arguments.config
    network_debug = arguments.debug
    #torch.cuda.set_device(arguments.local_rank)
    #torch.distributed.init_process_group(backend='nccl')

    # Load options
    json_opts = json_file_to_pyobj(json_filename)
    train_opts = json_opts.training

    # Architecture type
    arch_type = train_opts.arch_type

    # Setup Dataset and Augmentation
    ds_class = get_dataset(arch_type)
    ds_path  = get_dataset_path(arch_type, json_opts.data_path)
    ds_transform = get_dataset_transformation(arch_type, opts=json_opts.augmentation)
    #os.environ["CUDA_VISIBLE_DEVICES"]=json_opts.model.cut_gpu
    os.environ["CUDA_LAUNCH_BLOCKING"]="1"
    # Setup the NN Model
    model = get_model(json_opts.model)
    #torch.distributed.init_process_group('nccl',init_method='tcp://localhost:16006',world_size=1,rank=0)
    #model.net=torch.nn.parallel.DistributedDataParallel(model.net,device_ids=[0,2,3],output_device="cuda:0")

    if network_debug:
        print('# of pars: ', model.get_number_parameters())
        print('fp time: {0:.3f} sec\tbp time: {1:.3f} sec per sample'.format(*model.get_fp_bp_time()))
        exit()

    # Setup Data Loader
    train_dataset = ds_class(ds_path, split='train_512',  transform=ds_transform['train'], preload_data=train_opts.preloadData,wh=json_opts.model.wh)
    valid_dataset = ds_class(ds_path, split='val_512',    transform=ds_transform['valid'], preload_data=train_opts.preloadData,wh=json_opts.model.wh)
    #test_dataset  = ds_class(ds_path, split='val_512',    transform=ds_transform['valid'], preload_data=train_opts.preloadData)
    train_loader = DataLoader(dataset=train_dataset, num_workers=train_opts.num_workers, batch_size=train_opts.batchSize, shuffle=True)
    valid_loader = DataLoader(dataset=valid_dataset, num_workers=train_opts.num_workers, batch_size=train_opts.batchSize, shuffle=False)
    #test_loader  = DataLoader(dataset=test_dataset,  num_workers=train_opts.num_workers, batch_size=train_opts.batchSize, shuffle=False)

    # Visualisation Parameters
    #visualizer = Visualiser(json_opts.visualisation, save_dir=model.save_dir)
    vis        = ViShow(json_opts.visualisation, models=model,x=torch.rand(2,1,512,512).cuda())

    error_logger = ErrorLogger()

    # Training Function
    model.set_scheduler(train_opts)

    #path="/home/linda/wzm/Attention-Gated-Networks/src/CT_att_3/CT_att_3/046_net_0.16918.pth"
    #save_model=torch.load(path)
    #new_model_dict=model.net.state_dict()
    #state_dict={k:v for k,v in save_model.items() if k in new_model_dict.keys()}
    #new_model_dict.update(state_dict)
    #model.net.load_state_dict(new_model_dict)

    for epoch in range(model.which_epoch, train_opts.n_epochs):
        epoch=epoch+model.which_epoch
        print('(epoch: %d, total # iters: %d)' % (epoch, len(train_loader)))
        # Training Iterations
        model.net.train()
        for epoch_iter, (images, labels) in tqdm(enumerate(train_loader, 1), total=len(train_loader)):
            # Make a training update
            #print("label shape",labels.shape,"im",images.shape)
            model.set_input(images, labels)
            model.optimize_parameters()
            #model.optimize_parameters_accumulate_grd(epoch_iter)

            # Error visualisation
            errors = model.get_current_errors()
            error_logger.update(errors, split='train')

        # Validation and Testing Iterations
        #for loader, split in zip([valid_loader, test_loader], ['validation', 'test']):
        #for loader, split in zip([valid_loader, test_loader], ['validation', 'test']):
        split='validation'
        model.net.eval()
        with torch.no_grad():
            for epoch_iter, (images, labels) in tqdm(enumerate(valid_loader, 1), total=len(valid_loader)):
                # Make a forward pass with the model
                model.set_input(images, labels)
                model.validate()
                # Error visualisation
                errors = model.get_current_errors()
                stats = model.get_segmentation_stats()
                error_logger.update({**errors, **stats}, split=split)
                #error_logger.update(errors,split=split)
                # Visualise predictions
                #visuals = model.get_current_visuals()
                #visualizer.display_current_results(visuals, epoch=epoch, save_result=False)

        # Update the plots
        for split in ['train', 'validation']:
            vis.plot_current_errors(epoch, errors=error_logger.get_errors(split), split_name=split)
            vis.print_current_errors(epoch, errors=error_logger.get_errors(split), split_name=split)

        # Save the model parameters
        if epoch % train_opts.save_epoch_freq == 0:
               model.save(epoch,loss=error_logger.get_errors("validation")["Seg_Loss"])

        error_logger.reset()
        # Update the model learning rate
        model.update_learning_rate()

    vis.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='CNN Seg Training Function')
    json_pth="/home/linda/wzm/Attention-Gated-Networks/configs/config_CT_two_seg.json"

    parser.add_argument('-c', '--config', help='training config file',default=json_pth, required=False)
    parser.add_argument('-d', '--debug', help='returns number of parameters and bp/fp runtime', default=False,action='store_true')
    parser.add_argument('--local_rank',type=int)
    args = parser.parse_args()

    train(args)
