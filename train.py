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
    import test_new_loss
    import augmentations
    from model import network
    from datasets.test_dataset import TestDataset
    from datasets.train_dataset import TrainDataset
    from datasets.target_dataset import TargetDataset, DomainAdaptationDataLoader
    from itertools import chain
    torch.backends.cudnn.benchmark = True  # Provides a speedup

    debug = False

    def print_ae_grad():
        print(model.autoencoder.encoder[0].weight.grad if (model.autoencoder.encoder[0].weight.grad != None) else None)
    
    def print_bb_grad():
        for name, child in model.backbone.named_children():
            if name == "6":
                for w in child.parameters():
                    print("Backbone grad layer 3")
                    print(w.grad[0][0][0])
                    break
            if name == "7":
                for w in child.parameters():
                    print("Backbone grad layer 4")
                    print(w.grad[0][0][0])  
                    break


    
    args = parser.parse_arguments()
    start_time = datetime.now()
    output_folder = f"logs/{args.save_dir}/{args.experiment_name}_{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
    commons.make_deterministic(args.seed)
    commons.setup_logging(output_folder, console="debug")
    logging.info(" ".join(sys.argv))
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {output_folder}")

    #### Model
    model = network.GeoLocalizationNet(args.backbone, args.fc_output_dim, domain_adaptation= args.domain_adaptation,backbone_path= args.backbone_path, aada = args.aada)

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
    if args.aada:
        target_dataset = TargetDataset(args.target_dataset_folder)
        domain_criterion = torch.nn.CrossEntropyLoss()
        autoencoder_criterion = torch.nn.MSELoss()


    model_parameters = chain(model.backbone.parameters(), model.aggregation.parameters(), model.discriminator.parameters() if args.domain_adaptation else [], model.autoencoder.parameters() if args.aada else [])
    model_optimizer = torch.optim.Adam(model_parameters, lr=args.lr)



    #### Datasets
    # Each group is treated as a different dataset
    groups = [TrainDataset(args, args.train_set_folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L,
                        current_group=n, min_images_per_class=args.min_images_per_class, preprocessing=args.preprocessing, base_preprocessing = args.base_preprocessing) for n in range(args.groups_num)]
    
    if args.pseudo_target_folder:
        pseudo_groups = [TrainDataset(args, args.pseudo_target_folder, M=args.M, alpha=args.alpha, N=args.N, L=args.L,
                        current_group=n, min_images_per_class=args.min_images_per_class, preprocessing=args.preprocessing, base_preprocessing = args.base_preprocessing,
                        pseudo_target=True
                        ) for n in range(args.groups_num)]

    # Each group has its own classifier, which depends on the number of classes in the group
    if args.loss == "cosface": 
        classifiers = [cosface_loss.MarginCosineProduct(args.fc_output_dim, len(group)) for group in groups]
    elif args.loss == "sphereface":
        classifiers = [sphereface_loss.SphereFace(args.fc_output_dim, len(group)) for group in groups]
    elif args.loss == "arcface":
        classifiers = [arcface_loss.ArcFace(args.fc_output_dim, len(group)) for group in groups]
    else:
        logging.debug("No valid loss, please try again typing 'cosface', 'sphereface' or 'arcface'")

    classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifiers_lr) for classifier in classifiers]

    logging.info(f"Using {len(groups)} groups")
    logging.info(f"The {len(groups)} groups have respectively the following number of classes {[len(g) for g in groups]}")
    logging.info(f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")

    val_ds = TestDataset(args.val_set_folder, positive_dist_threshold=args.positive_dist_threshold)
    test_ds = TestDataset(args.test_set_folder, queries_folder="queries_v1",
                        positive_dist_threshold=args.positive_dist_threshold)
    
    if args.test_all:
        #to test on all the test set at the end of training
        logging.info(f"Testing all!")
        tokyo_xs_test_ds = TestDataset(args.tokyo_xs_dataset_folder, queries_folder="queries_v1",
                        positive_dist_threshold=args.positive_dist_threshold)
        
        tokyo_night_test_ds = TestDataset(args.tokyo_xs_dataset_folder, queries_folder="night/",
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

    apply_aug = False
    if args.augmentation_device == "cuda":
        apply_aug = True
        if args.augmentation_type == "brightness":
            augType = augmentations.DeviceAgosticAdjustBrightness(args.reduce_brightness)
        elif args.augmentation_type == "contrast":
            augType = augmentations.DeviceAgnosticContrast(args.increase_contrast)
        elif args.augmentation_type == "saturation":
            augType = augmentations.DeviceAgosticAdjustSaturation(args.decrease_saturation)
        elif args.augmentation_type == "bcs":
            augType = augmentations.DeviceAgosticAdjustBrightnessContrastSaturation(args.reduce_brightness,args.increase_contrast, args.decrease_saturation)
        elif args.augmentation_type == "colorjitter":
            augType = augmentations.DeviceAgnosticColorJitter(brightness=args.brightness,
                                                        contrast=args.contrast,
                                                        saturation=args.saturation,
                                                        hue=args.hue)
        elif args.augmentation_type == "none":
            apply_aug = False
        else:
            logging.debug("No valid augmentation, please try again typing 'brightness', 'contrast', 'saturation', 'bcs', 'colorjitter' or 'none'")
            exit()

    if apply_aug:
        gpu_augmentation = T.Compose([
            augType,
            augmentations.DeviceAgnosticRandomResizedCrop([512, 512],
                                                            scale=[1-args.random_resized_crop, 1]),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        gpu_augmentation = T.Compose([
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

    for epoch_num in range(start_epoch_num, args.epochs_num):
        
        #### Train
        epoch_start_time = datetime.now()
        # Select classifier and dataloader according to epoch
        current_group_num = epoch_num % args.groups_num
        classifiers[current_group_num] = classifiers[current_group_num].to(args.device)
        util.move_to_device(classifiers_optimizers[current_group_num], args.device)
        # setup the dataloader
        batch_size = args.batch_size
        if args.pseudo_target_folder:
            batch_size = int(batch_size / 2)
        
        

        dataloader = commons.InfiniteDataLoader(groups[current_group_num],
                                                pseudo_dataset = pseudo_groups[current_group_num] if args.pseudo_target_folder else None,
                                                 num_workers=args.num_workers,
                                                batch_size=batch_size, shuffle=True,
                                                pin_memory=(args.device == "cuda"), drop_last=True,
                                                )
        
       


        if args.domain_adaptation or args.aada:

            

            da_dataloader = DomainAdaptationDataLoader(groups[current_group_num], target_dataset,
                                                       pseudo_dataset= 
                                                       pseudo_groups[current_group_num] if args.pseudo_target_folder else None,
                                                       
                                                       pseudo=args.pseudo_da, num_workers=args.num_workers, 
                                                        batch_size = 16, shuffle=True,
                                                        pin_memory=(args.device == "cuda"), drop_last=True)
            
        dataloader_iterator = iter(dataloader)
        
        model = model.train()
        #list of epoch losses. At the end the mean will be computed
        epoch_losses = np.zeros((0, 1), dtype=np.float32)
        for iteration in tqdm(range(args.iterations_per_epoch), ncols=100):
            images, targets = next(dataloader_iterator)

            images, targets = images.to(args.device), targets.to(args.device)
            
            if args.domain_adaptation or args.aada:
                da_images, da_targets = next(da_dataloader)
                da_images, da_targets = da_images.to(args.device), da_targets.to(args.device)

            if args.augmentation_device == "cuda":
                images = gpu_augmentation(images)
                if args.domain_adaptation or args.aada:
                    da_images = target_augmentation(da_images)

                   
            
            model_optimizer.zero_grad()
            classifiers_optimizers[current_group_num].zero_grad()

            
            
            if not args.use_amp16:
                #Get descriptors from the model (ends with fc and normalization)
                descriptors = model(images)
                #Gets the output, that is the cosine similarity between the descriptors and the weights of the classifier
                output = classifiers[current_group_num](descriptors, targets)
                #Applies the softmax loss
                
                loss = criterion(output, targets)
                loss.backward()
                                                   
                #append the loss to the epoch losses

                
                da_loss = 0
                enc_loss = 0
                if args.domain_adaptation and not args.aada:
                    da_output = model(da_images, grl=True)
                    da_loss = criterion(da_output, da_targets)
                    (da_loss * args.grl_loss_weight).backward()
                    da_loss = (da_loss * args.grl_loss_weight).item()
                    del da_output, da_images

                if args.aada:
                    """
                    python train.py --dataset_folder small --groups_num 1 --epochs_num 3 --device cpu --target_dataset_folder tokyo-night --pseudo_target_folder small_night --aada True
                    """
                

                    features_source, features_target, enc_output_source, enc_output_target = model(da_images, aada=True, targets = da_targets)
                    enc_loss_source = autoencoder_criterion(enc_output_source, features_source)
                    enc_loss_target = autoencoder_criterion(enc_output_target, features_target)
                    
                    

                    

                    #aada on backbone loss pass
                    (args.aada_loss_weight * enc_loss_target).backward(retain_graph=True)
                    model.save_bb_grad()
                    model.autoencoder.encoder.zero_grad()
                    model.autoencoder.decoder.zero_grad()
                    
                    #aada on autoencoder loss pass
                    model.backbone.zero_grad()
                    model.aggregation.zero_grad()
                  
                    enc_loss = enc_loss_source + torch.max(torch.zero_(enc_loss_target), args.aada_m - enc_loss_target)
                    (enc_loss).backward()
                    enc_loss = enc_loss.item()
                    model.load_bb_grad()
                    

                # epoch_losses = np.append(epoch_losses, loss.item() + da_loss)
                epoch_losses = np.append(epoch_losses, loss.item() + da_loss + enc_loss)
                del loss, output, images, da_loss, enc_loss
                #optimize the parameters
                model_optimizer.step()
                #optimize the parameters of the classifier
                classifiers_optimizers[current_group_num].step()
            #todo: add domain adaptation here
            else:  # Use AMP 16
                with torch.cuda.amp.autocast():
                    descriptors = model(images)
                    output = classifiers[current_group_num](descriptors, targets)
                    loss = criterion(output, targets)
                scaler.scale(loss).backward()
                if args.domain_adaptation:
                    da_loss = 0
                    with torch.cuda.amp.autocast():
                        da_output = model(da_images, grl=True)
                        da_loss = criterion(da_output, da_targets)
                    scaler.scale((da_loss * args.grl_loss_weight)).backward()
                    del da_output, da_images

                if args.aada:
                    enc_loss = 0
                    with torch.cuda.amp.autocast():
                        images_source = da_images[da_targets==0, :, :, :]
                        images_target = da_images[da_targets==1, :, :, :]
                        enc_output_source, enc_output_target = model(da_images, aada=True, images_source=images_source, images_target=images_target)
                        
                        
                        #CE loss pass
                        da_loss = criterion(da_output, da_targets)
                        scaler.scale(da_loss).backward(retain_graph=True)
                        

                        #AE loss pass
                        enc_loss_source = autoencoder_criterion(enc_output_source, images_source)
                        enc_loss_target = autoencoder_criterion(enc_output_target, images_target)
                        enc_loss = enc_loss_source + max(0, args.aada_m - enc_loss_target)
                        scaler.scale(enc_loss).backward(retain_graph=True)
                        scaler.scale(enc_loss_target).backward(retain_graph=True)
                        
                        del da_output, da_images, enc_output_source, enc_output_target, images_source, images_target
                epoch_losses = np.append(epoch_losses, loss.item() + (da_loss * args.grl_loss_weight) + enc_loss ) 
                del loss, output, images, da_loss, enc_loss
                scaler.step(model_optimizer)
                scaler.step(classifiers_optimizers[current_group_num])
                scaler.update()
        
        classifiers[current_group_num] = classifiers[current_group_num].cpu()
        util.move_to_device(classifiers_optimizers[current_group_num], "cpu")
        
        logging.debug(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, "
                    f"loss = {epoch_losses.mean():.4f}")
        
        #### Evaluation
        recalls, recalls_str,_ = test.test(args, val_ds, model)
        logging.info(f"Epoch {epoch_num:02d} in {str(datetime.now() - epoch_start_time)[:-7]}, {val_ds}: {recalls_str[:20]}")
        is_best = recalls[0] > best_val_recall1
        best_val_recall1 = max(recalls[0], best_val_recall1)
        # Save checkpoint, which contains all training parameters
        util.save_checkpoint({
            "epoch_num": epoch_num + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": model_optimizer.state_dict(),
            "classifiers_state_dict": [c.state_dict() for c in classifiers],
            "optimizers_state_dict": [c.state_dict() for c in classifiers_optimizers],
            "best_val_recall1": best_val_recall1
        }, is_best, output_folder)


    logging.info(f"Trained for {epoch_num+1:02d} epochs, in total in {str(datetime.now() - start_time)[:-7]}")

    #### Test best model on test set v1
    best_model_state_dict = torch.load(f"{output_folder}/best_model.pth")
    model.load_state_dict(best_model_state_dict)

    logging.info(f"Now testing on the test set: {test_ds}")
    recalls, recalls_str,_ = test.test(args, test_ds, model)
    logging.info(f"{test_ds}: {recalls_str}")
    

    
    
    if args.test_all:  
        recalls, recalls_str,tokyo_xs_db_descriptors = test.test(args, tokyo_xs_test_ds, model)
        logging.info(f"{tokyo_xs_test_ds}: {recalls_str}")
        recalls, recalls_str,_ = test.test(args, tokyo_night_test_ds, model, db_descriptors = tokyo_xs_db_descriptors)
        logging.info(f"{tokyo_night_test_ds}: {recalls_str}")

    logging.info("Experiment finished (without any errors)")

