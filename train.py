# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from fuse_detection.trainer import trainer
from detectron2.engine import default_argument_parser, launch
if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    # args.config_file = '/home/ql-b423/Desktop/TXH/VOS/MaskPrototypical/instances/configs/mask_rcnn_X_101_32x8d_FPN_3x.yaml'

    ######################## only need to change here
    root_davis = '/home/ql-b423/sda/TXH/dataset/davis/'
    config_name = 'cascade_mask_rcnn_R_50_FPN_3x.yaml'
    learning_rate = 0.001
    max_iter = 10000
    steps = (7500,9000)
    warmup_iter = 2000
    batch_size = 4
    gpu_num  = 1
    checkpoint_period = 1000 # save weight per 1000

    pretrain_model = 'cascade.pkl'
    # if not ues pretrain model
    # pretrain_model = None
    MASK_ON = False

    args.eval_only = True
    # specify the mdoel name for evaluation, only need provide the file name
    args.best_model_name = 'model_0009999.pth'

    # if want to get visual result, change to True
    visual = True
    visual_threshold = 0.5
    ##############################

    args.num_gpus = gpu_num
    args.config_file = 'config/'+ config_name
    if pretrain_model is None:
        pretrain_path = None
    else:
        pretrain_path = 'pretrain/' + pretrain_model
    args.opts = {'root_davis':root_davis,'learning_rate':learning_rate,'max_iter':max_iter,'batch_size':batch_size,
                 'pretrain_path':pretrain_path,'warmup_iter':warmup_iter,'MASK_ON':MASK_ON,'steps':steps,'visual':visual,
                 'vusual_threshold':visual_threshold,'checkpoint_period':checkpoint_period}

    print("Command Line Args:", args)

    launch(
        trainer,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,visual,visual_threshold),
    )