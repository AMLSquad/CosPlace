

def add_and_save_in_set(set, descriptor, filename):
    if filename == "SKIP":
        return set
    if filename in set:
        return set
    set.add(filename)
    torch.save(descriptor, "compact/" + filename)
    return set

if __name__ == "__main__":
    import sys
    import torch
    import logging
    import numpy as np
    from tqdm import tqdm
    import multiprocessing
    from datetime import datetime
    import torchvision.transforms as T
    import test
    import util
    import my_parser as parser
    import commons
    import sphereface_loss
    import cosface_loss
    import arcface_loss 
    import augmentations
    from model import network
    from datasets.test_dataset import TestDataset
    from datasets.train_dataset import TrainDataset
    from datasets.target_dataset import TargetDataset, DomainAdaptationDataLoader
    from torch.utils.data import DataLoader
    from itertools import chain

    torch.backends.cudnn.benchmark = True  # Provides a speedup





    args = parser.parse_arguments()
    
    start_time = datetime.now()
    output_folder = f"logs/{args.save_dir}/{args.experiment_name}_{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
    commons.make_deterministic(args.seed)
    commons.setup_logging(output_folder, console="debug")
    logging.info(" ".join(sys.argv))
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {output_folder}")
    
    #### Model
    model = network.GeoLocalizationNet(args.backbone, args.fc_output_dim, domain_adaptation= args.domain_adaptation)

    logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")

    if args.resume_model is not None:
        logging.debug(f"Loading model from {args.resume_model}")
        model_state_dict = torch.load(args.resume_model)
        model.load_state_dict(model_state_dict)

    # set model to train mode
    model = model.to(args.device).train()
    #### Optimizer
    criterion = torch.nn.CrossEntropyLoss()
    # Remove the domain classifier parameters from the model parameters


    if args.domain_adaptation:
        target_dataset = TargetDataset(args.target_dataset_folder)
        domain_criterion = torch.nn.CrossEntropyLoss()


    model_parameters = chain(model.backbone.parameters(), model.aggregation.parameters(), model.discriminator.parameters() if args.domain_adaptation else [])
    model_optimizer = torch.optim.Adam(model_parameters, lr=args.lr)


        

    #### Datasets
    # Each group is treated as a different dataset
    groups = [TrainDataset(args, args.train_set_folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L,
                        current_group=n, min_images_per_class=args.min_images_per_class, preprocessing=args.preprocessing) for n in range(args.groups_num)]


    # Each group has its own classifier, which depends on the number of classes in the group
    if args.loss == "cosface": 
        classifiers = [cosface_loss.MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups]
    elif args.loss == "sphereface":
        classifiers = [sphereface_loss.SphereFace(args.fc_output_dim, len(group)) for group in groups]
    elif args.loss == "arcface":
        classifiers = [arcface_loss.MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups]
    else:
        logging.debug("No valid loss, please try again typing 'cosface', 'sphereface' or 'arcface'")
        exit
    classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in classifiers]

    logging.info(f"Using {len(groups)} groups")
    logging.info(f"The {len(groups)} groups have respectively the following number of classes {[len(g) for g in groups]}")
    logging.info(f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")

    val_ds = TestDataset(args.val_set_folder, positive_dist_threshold=args.positive_dist_threshold)
    test_ds = TestDataset(args.test_set_folder, queries_folder="queries_v1",
                        positive_dist_threshold=args.positive_dist_threshold)
    logging.info(f"Validation set: {val_ds}")
    logging.info(f"Test set: {test_ds}")

    #### Resume
    if args.resume_train:
        model, model_optimizer, classifiers, classifiers_optimizers, best_val_recall1, start_epoch_num = \
            util.resume_train(args, output_folder, model, model_optimizer, classifiers, classifiers_optimizers)
        model = model.to(args.device)
        epoch_num = start_epoch_num - 1
        logging.info(f"Resuming from epoch {start_epoch_num} with best R@1 {best_val_recall1:.1f} from checkpoint {args.resume_train}")
    else:
        best_val_recall1 = start_epoch_num = 0

    #### Train / evaluation loop
    logging.info("Start training ...")
    logging.info(f"There are {len(groups[0])} classes for the first group, " +
                f"each epoch has {args.iterations_per_epoch} iterations " +
                f"with batch_size {args.batch_size}, therefore the model sees each class (on average) " +
                f"{args.iterations_per_epoch * args.batch_size / len(groups[0]):.1f} times per epoch")


    if args.augmentation_device == "cuda":
        gpu_augmentation = T.Compose([
                augmentations.DeviceAgnosticColorJitter(brightness=args.brightness,
                                                        contrast=args.contrast,
                                                        saturation=args.saturation,
                                                        hue=args.hue),
                augmentations.DeviceAgnosticRandomResizedCrop([512, 512],
                                                            scale=[1-args.random_resized_crop, 1]),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        target_augmentation = T.Compose([
            augmentations.DeviceAgnosticRandomResizedCrop([512, 512],
                                                            scale=[1-args.random_resized_crop, 1]),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


    if args.use_amp16:
        scaler = torch.cuda.amp.GradScaler()
    """
    images_set = set()
    print(groups[0].classes_ids)
    for idx, class_id in enumerate(groups[0].classes_ids):
        for j, image in enumerate(groups[0].images_per_class[class_id]):
            image, targets, filename, _ = groups[0].__getitem__(idx, j)
            image = image.unsqueeze(0)
            image  = image.to(args.device)
            descriptor = model(image)
            torch.save(descriptor, "compact/" + filename)
    """
    for idx, class_id in enumerate(val_ds.images_paths):
        image,  _, name= val_ds.__getitem__(idx)
        image = image.unsqueeze(0)
        image  = image.to(args.device)

        descriptor = model(image)
        torch.save(descriptor, "compact/val/" + name)

        
"""
    for epoch_num in range(start_epoch_num, args.epochs_num):
        
        #### Train
        epoch_start_time = datetime.now()
        # Select classifier and dataloader according to epoch
        current_group_num = epoch_num % args.groups_num
        print(current_group_num)
        classifiers[current_group_num] = classifiers[current_group_num].to(args.device)
        util.move_to_device(classifiers_optimizers[current_group_num], args.device)
        # setup the dataloader
        dataloader = commons.InfiniteDataLoader(groups[current_group_num], num_workers=args.num_workers,
                                                batch_size=args.batch_size, shuffle=True,
                                                pin_memory=(args.device == "cuda"), drop_last=True)
        
        if args.domain_adaptation:
            da_dataloader = DomainAdaptationDataLoader(groups[current_group_num], target_dataset,num_workers=args.num_workers,
                                                        batch_size = 24, shuffle=True,
                                                        pin_memory=(args.device == "cuda"), drop_last=True)
        dataloader_iterator = iter(dataloader)
        
        model = model.train()
        #list of epoch losses. At the end the mean will be computed
        epoch_losses = np.zeros((0, 1), dtype=np.float32)
        for iteration in tqdm(range(args.iterations_per_epoch), ncols=100):
            images, targets, filename, _ = next(dataloader_iterator)
            
            images, targets = images.to(args.device), targets.to(args.device)
            
            if args.domain_adaptation:
                da_images, da_targets, da_filename = next(da_dataloader)
                da_images, da_targets = da_images.to(args.device), da_targets.to(args.device)

            if args.augmentation_device == "cuda":
                images = gpu_augmentation(images)
            
            model_optimizer.zero_grad()
            classifiers_optimizers[current_group_num].zero_grad()

            
            
            if not args.use_amp16:
                #Get descriptors from the model (ends with fc and normalization)
                descriptors = model(images)
                for idx, d in enumerate(descriptors):
                    if not filename[idx] in images_set:
                        add_and_save_in_set(images_set, descriptors[idx], filename[idx])

                #Gets the output, that is the cosine similarity between the descriptors and the weights of the classifier
                
                #append the loss to the epoch losses

                
                da_loss = 0
                if args.domain_adaptation:
                    da_output = model(da_images, grl=True)
                    for idx, d in enumerate(da_output):
                        add_and_save_in_set(images_set, da_output[idx], da_filename[idx])
                        
                    
                #optimize the parameters
                #optimize the parameters of the classifier
            #todo: add domain adaptation here
            else:  # Use AMP 16
                with torch.cuda.amp.autocast():
                    descriptors = model(images)
                    for idx, d in enumerate(descriptors):
                        add_and_save_in_set(images_set, descriptors[idx], filename[idx])

                if args.domain_adaptation:
                    da_loss = 0
                    with torch.cuda.amp.autocast():
                        descriptors = model(da_images, grl=True)
                        for idx, d in enumerate(descriptors):
                            add_and_save_in_set(images_set, da_output[idx], da_filename[idx])
                    del da_output, da_images
                
        
        


    logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

    

    logging.info("Experiment finished (without any errors)")


"""
