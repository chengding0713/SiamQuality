import argparse

def parse_args():
    
    parser = argparse.ArgumentParser(description='PPG Training'
    )
    # export directory, training and val datasets, test datasets
    parser.add_argument('--LOG_DIR', 
        default='logs',
        help = 'Path to log folder'
    )
    parser.add_argument('--baseline', 
        default='simsiam', # byol, simclr, swav, moco, simsiam
        help = 'Baseline used'
    )
    
    parser.add_argument('--encoder', 
        default ='ResNet152', # ResNet50, ResNet101, ResNetSwav50, ResNetSwav101 -> Naming matters. See the models/resnet.py for naming of the encoders
        help = 'Encoder used to extract the features'
    )

    parser.add_argument('--input_pair_path',
        default = 'G:/UCSF_waveform_PPG_30second_quality_pair_binary.npy',
        help = 'path to the input file (.npy)'
    )

    parser.add_argument('--embedding-size', default = 256, type = int,
        dest = 'sz_embedding',
        help = 'Size of embedding that is appended to backbone model.'
    )
    parser.add_argument('--batch-size', default = 256, type = int,
        dest = 'sz_batch',
        help = 'Number of samples per batch.'
    )
    parser.add_argument('--epochs', default = 500, type = int,
        dest = 'nb_epochs',
        help = 'Number of training epochs.'
    )
    parser.add_argument('--gpu-id', default = 0, type = int,
        help = 'ID of GPU that is used for training.'
    )
    parser.add_argument('--workers', default = 0, type = int,
        dest = 'nb_workers',
        help = 'Number of workers for dataloader.'
    )
    parser.add_argument('--model', default = 'backbone', #classifier, regressor, backbone
        help = 'Model for training'
    )
    parser.add_argument('--loss', default = '', #CrossEntropy, MseLoss, NegativeCosine
        help = 'Criterion for training'
    )
    parser.add_argument('--optimizer', default = 'sgd',
        help = 'Optimizer setting'
    )
    parser.add_argument('--lr', default = 1e-4, type =float,
        help = 'Learning rate setting'
    )
    parser.add_argument('--weight-decay', default = 1e-4, type =float,
        help = 'Weight decay setting'
    )
    parser.add_argument('--lr-decay-step', default = 10, type =int,
        help = 'Learning decay step setting'
    )
    parser.add_argument('--lr-decay-gamma', default = 0.5, type =float,
        help = 'Learning decay gamma setting'
    )
    parser.add_argument('--alpha', default = 32, type = float,
        help = 'Scaling Parameter setting'
    )
    parser.add_argument('--mrg', default = 0.1, type = float,
        help = 'Margin parameter setting'
    )
    parser.add_argument('--IPC', type = int,
        help = 'Balanced sampling, images per class'
    )
    parser.add_argument('--warm', default = 50, type = int,
        help = 'Warmup training epochs'
    )
    parser.add_argument('--bn-freeze', default = 1, type = int,
        help = 'Batch normalization parameter freeze'
    )
    parser.add_argument('--l2-norm', default = 1, type = int,
        help = 'L2 normlization'
    )
    parser.add_argument('--remark', default = '',
        help = 'Any remark'
    )
    parser.add_argument('--signal_length', default = 1200, type = int,
        help = 'signal length'
    )
    parser.add_argument('--sample_size', default = 1000, type = int,
        help = 'sample size of the signal'
    )
    parser.add_argument('--num_classes', default = 2, type = int,
        help = 'classification classes'
    )
    parser.add_argument('--pretrained_weights', default = r"G:\contrastive_152\logs\logs_1703906700_simsiam\backbone__embedding256_alpha32_mrg0.1_sgd_lr0.0001_batch256/simsiam_backbone_256__1.pth",
        type = str,
        help = 'path to the pretrained weight of the baselines'
    )
    
    parser.add_argument('--finetuned_weights', default = "",
        type = str,
        help = 'path to the finetuned weight for the downstream task'
    )
    
    return parser.parse_args()

