from args_config import PARSER_CONFIG
from parser import Parser
from extract_features import run_feature_extraction
from train import train
from test import test

try:
    parser = Parser(prog='VQA',
                    description='VQA model training and testing program',
                    )

    args = parser.get_args(
        PARSER_CONFIG
    )
    print([(key,value) for key, value in vars(args).items()])

except ValueError as e:
    print(e)

if __name__ == '__main__':
    if args.run_extraction:
        run_feature_extraction(args)

    if args.run_training:
        train(config=args)

    if args.run_testing:
        test(config=args)