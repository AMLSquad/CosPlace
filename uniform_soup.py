from datasets.test_dataset import TestDataset
import test
from model import network
import torch
from datetime import datetime
import my_parser as parser
import os
import commons
import logging
import sys

def load_model(model_path,args):
    model = network.GeoLocalizationNet(args.backbone, args.fc_output_dim)
    model_state_dict = torch.load(model_path)
    model.load_state_dict(delete_discriminator_layer(model_state_dict))
    return model


def uniform_soup(models_list,  args):
    all_params = list(models_list[0].state_dict().keys())
    soup = {}
    for param in all_params:
        soup[param] = sum([m.state_dict()[param].clone() for m in models_list]) / len(models_list)
    agg_model = network.GeoLocalizationNet(args.backbone, args.fc_output_dim)
    agg_model.load_state_dict(soup)
    val_ds = TestDataset(args.val_set_folder, positive_dist_threshold=args.positive_dist_threshold)
    agg_model = agg_model.to(args.device)
    agg_model = agg_model.eval()
    _, recalls_str,_ = test.test(args, val_ds, agg_model)
    logging.info(f"{val_ds}: {recalls_str}")
    torch.save(agg_model.state_dict(),f"soup.pth")


def greedy_soup(models_list, args):
    sorted_models = []
    val_ds = TestDataset(args.test_set_folder, queries_folder=args.test_queries_folder,
                        positive_dist_threshold=args.positive_dist_threshold)

    for model in models_list:
        model = model.to(args.device)
        model = model.eval()       
        recalls, _ = test.test(args, val_ds, model)
        sorted_models.append((model, recalls[0]))

    sorted_models.sort(key=compare, reverse=True)
    greedy_soup_ingredients = [sorted_models[0][0]]
    greedy_soup_params = sorted_models[0][0].state_dict()
    num_ingredients = len(greedy_soup_ingredients)
    best_val_rec = sorted_models[0][1]
    for i in range(1,len(sorted_models)):
        new_ingredient_params = sorted_models[i][0].state_dict()
        potential_greedy_soup_params = {
                k : greedy_soup_params[k].clone() * (num_ingredients / (num_ingredients + 1.)) + 
                    new_ingredient_params[k].clone() * (1. / (num_ingredients + 1))
                for k in new_ingredient_params
            }
        new_model = network.GeoLocalizationNet(args.backbone, args.fc_output_dim)
        new_model.load_state_dict(potential_greedy_soup_params)
        new_model = new_model.to(args.device)
        new_model = new_model.eval()
        new_recall, _ = test.test(args, val_ds, new_model)
        print(f'Potential greedy soup val acc {new_recall[0]}, best so far {best_val_rec}.')
        if new_recall[0] > best_val_rec:
            greedy_soup_ingredients.append(sorted_models[i][0])
            best_val_rec = new_recall[0]
            greedy_soup_params = potential_greedy_soup_params
            print(f'Adding to soup.')
    
    torch.save(greedy_soup_params, f"soups_output/{args.experiment_name}/soup.pth") 

    
def compare(m1):
    return m1[1]

def delete_discriminator_layer(model):
    if "discriminator.1.weight" in model:
            del model["discriminator.1.weight"]
            del model["discriminator.1.bias"]
            del model["discriminator.3.weight"]
            del model["discriminator.3.bias"]
            del model["discriminator.5.weight"]
            del model["discriminator.5.bias"]
    return model


if __name__ == "__main__":
    args = parser.parse_arguments(is_training=True)

    output_folder = f"soups_ouput/{args.experiment_name}/"
    commons.setup_logging(output_folder, console="debug")
    logging.info(" ".join(sys.argv))
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {output_folder}")


    models_path= [path for path in os.listdir(args.soup_folder)]
    models_list = [load_model(args.soup_folder + model_path, args) for model_path in models_path]
    
    if args.greedy_soup:
        greedy_soup(models_list, args)
    if args.uniform_soup:
        uniform_soup(models_list, args)


