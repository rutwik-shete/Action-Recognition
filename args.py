import argparse

def argument_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--dataset_path", type=str, default="./dataset", help="path to data directory"
    )
    
    return parser