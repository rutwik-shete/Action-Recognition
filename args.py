import argparse

def argument_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--home_path", type=str, default="../", help="path to home directory"
    )

    parser.add_argument(
        "--dataset_path", type=str, default="../HMDB_simp", help="path to data directory"
    )

    parser.add_argument(
        "--save_dir", type=str, default="../Logs", help="path to save logs and checkpoints"
    )

    parser.add_argument(
        "--resume", type=str, default="../Logs", help="load check point from"
    )

    parser.add_argument(
        "--block_size", type=int, default="8", help="frame block size"
    )

    parser.add_argument(
        "--train_batch_size", type=int, default="64", help="train dataloader batch size"
    )    

    parser.add_argument(
        "--val_batch_size", type=int, default="64", help="validation dataloader batch size"
    )

    parser.add_argument(
        "--test_batch_size", type=int, default="64", help="test dataloader batch size"
    )

    parser.add_argument(
        "--epochs", type=int, default="10", help="epoch for training"
    )

    parser.add_argument(
        "--eval_freq", type=int, default="5", help="run test on validation set after every _ epochs"
    )

    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate for model"
    )

    parser.add_argument(
        "--model", type=str, default="timesformer400", help="model to pick for train/test"
    )

    parser.add_argument(
        "-t",
        "--target-names",
        type=str,
        required=True,
        nargs="+",
        help="target dataset for testing(delimited by space)",
    )

    parser.add_argument(
        "-s",
        "--source-names",
        type=str,
        required=True,
        nargs="+",
        help="source dataset for training(delimited by space)",
    )

    parser.add_argument(
        "--attn_dim", type=int, default=64, help="attention dimension"
    )  
         
    parser.add_argument(
        "--dropout", type=float, default=0.3, help="dropout rate"
    ) 

    parser.add_argument(
        '--skip_attention', action='store_true', default=False
        )

    parser.add_argument(
        "--run_name", type=str, required=True, help="run name for W&B"
    ) 
    return parser