import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    parser.add_argument('--local-rank', type=int, default=0, help='Local rank for distributed training')

    # textVQA
    parser.add_argument("--textVQA_image_dir", type=str, default="")
    parser.add_argument("--textVQA_ann_path", type=str, default="")

    # docVQA
    parser.add_argument("--docVQA_image_dir", type=str, default="")
    parser.add_argument("--docVQA_ann_path", type=str, default="")

    # docVQATest
    parser.add_argument("--docVQATest_image_dir", type=str, default="")
    parser.add_argument("--docVQATest_ann_path", type=str, default="")

    # result path
    parser.add_argument("--answer_path", type=str, default="./answers-new")

    # eval
    parser.add_argument(
        "--eval_textVQA",
        action="store_true",
        default=False,
        help="Whether to evaluate on textVQA."
    )
    parser.add_argument(
        "--eval_docVQA",
        action="store_true",
        default=False,
        help="Whether to evaluate on docVQA."
    )
    parser.add_argument(
        "--eval_docVQATest",
        action="store_true",
        default=False,
        help="Whether to evaluate on docVQA."
    )
    
    parser.add_argument(
        "--eval_all",
        action="store_true",
        default=False,
        help="Whether to evaluate all datasets"
    )

    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--model_path", type=str, default="")

    parser.add_argument("--generate_method", type=str, default="", help="generate with interleave or not.")

    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument('--batchsize', type=int, default=1, help='Batch size for processing.')

    parser.add_argument("--ckpt", type=str, default=None)

    args = parser.parse_args()
    return args