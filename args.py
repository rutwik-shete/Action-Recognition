import argparse

def argument_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--dataset_path", type=str, default="./dataset", help="path to data directory"
    )

    parser.add_argument(
        "--block_size", type=int, default="8", help="frame block size"
    )

    parser.add_argument(
        "--train_batch_size", type=int, default="64", help="train dataloader batch size"
    )    

    parser.add_argument(
        "--val_batch_size", type=int, default="70", help="validation dataloader batch size"
    )

    parser.add_argument(
        "--test_batch_size", type=int, default="70", help="test dataloader batch size"
    )

    parser.add_argument(
        "--epochs", type=int, default="10", help="epoch for training"
    )

    return parser