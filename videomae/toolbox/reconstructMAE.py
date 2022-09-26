import torch
import argparse

from timm.models import create_model

def parse_terminal_argument():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", type=str,)
    parser.add_argument("--mask-ratio", type=int,)
    parser.add_argument("--output-dir", type=str,)
    parser.add_argument("--model", type=str,)


    return parser.parse_args()


def preprocess_input(args):

    pass


def visualize(args, recons_img):

    pass


def main():
    args = parse_terminal_argument()
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=0.1,
        drop_block_rate=None,
        decoder_depth=4,
    )

    model_input = preprocess_input(args)

    recons_img = model(model_input)

    visualize(args, recons_img)



if __name__ == "__main__":
    main()
