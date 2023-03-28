
import os
import argparse


def parse_arguments(is_training: bool = True):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # CosPlace Groups parameters
    parser.add_argument("--M", type=int, default=10, help="_")
    parser.add_argument("--alpha", type=int, default=30, help="_")
    parser.add_argument("--N", type=int, default=5, help="_")
    parser.add_argument("--L", type=int, default=2, help="_")
    parser.add_argument("--groups_num", type=int, default=1, help="_")
    parser.add_argument("--min_images_per_class", type=int, default=10, help="_")
    # Model parameters
    parser.add_argument("--backbone", type=str, default="resnet18",
                        choices=["vgg16", "resnet18", "resnet50", "resnet101", "resnet152", "resnet18_places", "resnet18_gldv2"], help="_")
    parser.add_argument("--fc_output_dim", type=int, default=512,
                        help="Output dimension of final fully connected layer")
    # Training parameters
    parser.add_argument("--use_amp16", action="store_true",
                        help="use Automatic Mixed Precision")
    parser.add_argument("--augmentation_device", type=str, default="cuda",
                        choices=["cuda", "cpu"],
                        help="on which device to run data augmentation")
    parser.add_argument("--batch_size", type=int, default=32, help="_")
    parser.add_argument("--epochs_num", type=int, default=3, help="_")
    parser.add_argument("--iterations_per_epoch", type=int, default=10000, help="_")
    parser.add_argument("--lr", type=float, default=0.00001, help="_")
    parser.add_argument("--classifiers_lr", type=float, default=0.01, help="_")
    # Data augmentation
    parser.add_argument("--brightness", type=float, default=0.7, help="_")
    parser.add_argument("--contrast", type=float, default=0.7, help="_")
    parser.add_argument("--hue", type=float, default=0.5, help="_")
    parser.add_argument("--saturation", type=float, default=0.7, help="_")
    parser.add_argument("--random_resized_crop", type=float, default=0.5, help="_")
    # Validation / test parameters
    parser.add_argument("--infer_batch_size", type=int, default=16,
                        help="Batch size for inference (validating and testing)")
    parser.add_argument("--positive_dist_threshold", type=int, default=25,
                        help="distance in meters for a prediction to be considered a positive")
    # Resume parameters
    parser.add_argument("--resume_train", type=str, default=None,
                        help="path to checkpoint to resume, e.g. logs/.../last_checkpoint.pth")
    parser.add_argument("--resume_model", type=str, default=None,
                        help="path to model to resume, e.g. logs/.../best_model.pth")
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="_")
    parser.add_argument("--seed", type=int, default=0, help="_")
    parser.add_argument("--num_workers", type=int, default=1, help="_")
    # Paths parameters
    parser.add_argument("--dataset_folder", type=str, default="small/",
                        help="path of the folder with train/val/test sets")
    parser.add_argument("--save_dir", type=str, default="default",
                        help="name of directory on which to save the logs, under logs/save_dir")
    parser.add_argument("--target_dataset_folder", type=str, default="tokyo_xs/test/night/",
                        help="path of the folder with images from the target for DA sets")
    parser.add_argument("--domain_adaptation", type=bool, default=False)
    parser.add_argument("--grl_loss_weight", type=float, default=0.1, help="Weight for GRL loss")
    parser.add_argument("--pseudo_target_folder", type=str, default="")
    parser.add_argument("--test_queries_folder", type=str, default="queries_v1")
    parser.add_argument("--experiment_name",type=str,default="")
    parser.add_argument("--loss",type=str, default="cosface")
    parser.add_argument("--preprocessing", type=str, default=False)
    parser.add_argument("--augmentation_type", type=str, default="colorjitter")
    parser.add_argument("--reduce_brightness", type=float, default=0.65)
    parser.add_argument("--increase_contrast", type=float, default=1.15)
    parser.add_argument("--decrease_saturation", type=float, default=0.85)
    parser.add_argument("--soup_folder", type=str, default="soup_models/")
    parser.add_argument("--base_preprocessing", type=bool, default=False)
    parser.add_argument("--uniform_soup", type=bool, default=False)
    parser.add_argument("--greedy_soup", type=bool, default=False)
    parser.add_argument("--test_tokyo_night", type=bool, default=False)
    
    parser.add_argument("--focal_loss", type=bool, default=False)
    parser.add_argument("--l_loss" , type=float, default=1)
    parser.add_argument("--backbone_path", type=str, default=None)
    args = parser.parse_args()
    
    

    if args.dataset_folder is None:
        try:
            args.dataset_folder = os.environ['SF_XL_PROCESSED_FOLDER']
        except KeyError:
            raise Exception("You should set parameter --dataset_folder or export " +
                            "the SF_XL_PROCESSED_FOLDER environment variable as such \n" +
                            "export SF_XL_PROCESSED_FOLDER=/path/to/sf_xl/processed")
    
    if not os.path.exists(args.dataset_folder):
        raise FileNotFoundError(f"Folder {args.dataset_folder} does not exist")
    
    if is_training:
        args.train_set_folder = os.path.join(args.dataset_folder, "train")
        if not os.path.exists(args.train_set_folder):
            raise FileNotFoundError(f"Folder {args.train_set_folder} does not exist")
        
        args.val_set_folder = os.path.join(args.dataset_folder, "val")
        if not os.path.exists(args.val_set_folder):
            raise FileNotFoundError(f"Folder {args.val_set_folder} does not exist")
    
    args.test_set_folder = os.path.join(args.dataset_folder, "test")
    if not os.path.exists(args.test_set_folder):
        raise FileNotFoundError(f"Folder {args.test_set_folder} does not exist")

    if args.loss not in ["cosface", "arcface", "sphereface", "new_loss"]:
        raise Exception("No valid loss for --loss arg, please try again typing 'cosface', 'sphereface' or 'arcface'")


    return args
