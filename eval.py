
if __name__ == "__main__":
    import sys
    import torch
    import logging
    import multiprocessing
    from datetime import datetime

    import test
    import my_parser as parser 
    import commons
    from model import network
    from datasets.test_dataset import TestDataset

    torch.backends.cudnn.benchmark = True  # Provides a speedup

    autoencoder_layer = ["autoencoder.encoder.0.weight", "autoencoder.encoder.0.bias", "autoencoder.encoder.2.weight", "autoencoder.encoder.2.bias", "autoencoder.encoder.4.weight", "autoencoder.encoder.4.bias", "autoencoder.decoder.0.weight", "autoencoder.decoder.0.bias", "autoencoder.decoder.2.weight", "autoencoder.decoder.2.bias", "autoencoder.decoder.4.weight", "autoencoder.decoder.4.bias"]

    args = parser.parse_arguments(is_training=False)
    start_time = datetime.now()
    output_folder = f"logs/{args.save_dir}/{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
    commons.make_deterministic(args.seed)
    commons.setup_logging(output_folder, console="info")
    logging.info(" ".join(sys.argv))
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {output_folder}")

    logging.info(f"There are {torch.cuda.device_count()} GPUs and {multiprocessing.cpu_count()} CPUs.")
    #### Model
    model = network.GeoLocalizationNet(args.backbone, args.fc_output_dim, backbone_path=args.backbone_path)

    if args.resume_model is not None:
        logging.info(f"Loading model from {args.resume_model}")
        model_state_dict = torch.load(args.resume_model)
        if args.domain_adaptation:
            del model_state_dict["discriminator.1.weight"]
            del model_state_dict["discriminator.1.bias"]
            del model_state_dict["discriminator.3.weight"]
            del model_state_dict["discriminator.3.bias"]
            del model_state_dict["discriminator.5.weight"]
            del model_state_dict["discriminator.5.bias"]
        if args.aada:
            for l in autoencoder_layer:
                del model_state_dict[l]



        model.load_state_dict(model_state_dict)
    else:
        logging.info("WARNING: You didn't provide a path to resume the model (--resume_model parameter). " +
                    "Evaluation will be computed using randomly initialized weights.")

    model = model.to(args.device)

    test_ds = TestDataset(args.test_set_folder, queries_folder=args.test_queries_folder,
                        positive_dist_threshold=args.positive_dist_threshold)

    recalls, recalls_str, descriptors = test.test(args, test_ds, model)
    logging.info(f"{test_ds}: {recalls_str}")

    if args.test_tokyo_night:
        test_ds = TestDataset(args.test_set_folder, queries_folder="night/",
                        positive_dist_threshold=args.positive_dist_threshold)
        recalls, recalls_str, _ = test.test(args, test_ds, model, descriptors)
        logging.info(f"{test_ds}: {recalls_str}")
    