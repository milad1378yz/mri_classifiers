import argparse
import yaml
from typing import Tuple


def parse_config(config_file: str) -> argparse.Namespace:
    """
    Parse the command-line arguments.

    :param config_file: str, path to the configuration file
    :return: argparse.Namespace object containing the parsed arguments

    This function takes a path to a configuration file and parses the command-line arguments. The configuration file is expected to be in YAML format. The parsed arguments are returned as an argparse.Namespace object.
    """
    # Load the YAML configuration file
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default=config["data_path"],
        help="path to the input data_path",
    )
    parser.add_argument(
        "--results_path",
        type=str,
        default=config["results_path"],
        help="path for the resluts",
    )

    parser.add_argument(
        "--do_augmentation",
        type=bool,
        default=config["do_augmentation"],
        help="do_augmentation",
    )
    parser.add_argument(
        "--do_generation",
        type=bool,
        default=config["do_generation"],
        help="do_generation using diffusion models",
    )

    parser.add_argument(
        "--convert_binary",
        type=bool,
        default=config["convert_binary"],
        help="converts labels to binary",
    )

    parser.add_argument(
        "--feature_selection",
        type=bool,
        default=config["feature_selection"],
        help="feature_selection",
    )
    parser.add_argument(
        "--num_feature",
        type=int,
        default=config["num_feature"],
        help="number_of_features",
    )

    parser.add_argument("--do_svm", type=bool, default=config["do_svm"], help="do_svm")
    parser.add_argument(
        "--max_iter", type=int, default=config["max_iter"], help="max_iter"
    )

    parser.add_argument("--do_knn", type=bool, default=config["do_knn"], help="do_knn")
    parser.add_argument(
        "--n_neighbors", type=int, default=config["n_neighbors"], help="n_neighbors"
    )

    parser.add_argument(
        "--do_random_forest",
        type=bool,
        default=config["do_random_forest"],
        help="do_random_forest",
    )
    parser.add_argument(
        "--n_estimators", type=int, default=config["n_estimators"], help="n_estimators"
    )
    parser.add_argument(
        "--max_depth",
        type=lambda x: int(x) if x.isdigit() else None,
        default=config["max_depth"],
        help="max_depth",
    )

    parser.add_argument("--do_mlp", type=bool, default=config["do_mlp"], help="do_mlp")
    parser.add_argument(
        "--hidden_layer_sizes",
        type=Tuple[int],
        default=config["hidden_layer_sizes"],
        help="hidden_layer_sizes",
    )
    parser.add_argument(
        "--activation", type=str, default=config["activation"], help="activation"
    )
    parser.add_argument(
        "--learning_rate_init",
        type=float,
        default=config["learning_rate_init"],
        help="learning_rate_init",
    )
    parser.add_argument(
        "--validation_fraction",
        type=float,
        default=config["validation_fraction"],
        help="validation_fraction",
    )

    parser.add_argument(
        "--do_logistic_regression",
        type=bool,
        default=config["do_logistic_regression"],
        help="do_logistic_regression",
    )

    parser.add_argument("--do_cnn", type=bool, default=config["do_cnn"], help="do_cnn")
    parser.add_argument(
        "--batch_size_cnn",
        type=int,
        default=config["batch_size_cnn"],
        help="batch_size_cnn",
    )
    parser.add_argument(
        "--num_epochs_cnn",
        type=int,
        default=config["num_epochs_cnn"],
        help="num_epochs_cnn",
    )
    parser.add_argument(
        "--learning_rate_cnn",
        type=float,
        default=config["learning_rate_cnn"],
        help="learning_rate of cnn",
    )

    parser.add_argument(
        "--do_ensemble_adaboost",
        type=bool,
        default=config["do_ensemble_adaboost"],
        help="do_ensemble_adaboost",
    )

    parser.add_argument(
        "--do_decision_tree",
        type=bool,
        default=config["do_decision_tree"],
        help="do_decision_tree",
    )
    parser.add_argument(
        "--max_depth_decision_tree",
        type=int,
        default=config["max_depth_decision_tree"],
        help="max_depth_decision_tree",
    )

    parser.add_argument(
        "--do_naive_bayes",
        type=bool,
        default=config["do_naive_bayes"],
        help="do_naive_bayes",
    )

    parser.add_argument("--do_vit", type=bool, default=config["do_vit"], help="do_vit")

    # Parse the command-line arguments
    args, _ = parser.parse_known_args()
    args_checker(args)

    return args


def args_checker(args):
    # check if validation_fraction is between 0 and 1
    assert (
        args.validation_fraction >= 0 and args.validation_fraction <= 1
    ), "validation_fraction should be between 0 and 1"

    # check that just cnn is true and no other classifier is true or cnn is false and at least one other classifier is true
    if args.do_cnn or args.do_vit:
        assert not any(
            [
                args.do_svm,
                args.do_knn,
                args.do_random_forest,
                args.do_mlp,
                args.do_logistic_regression,
                args.do_ensemble_adaboost,
                args.do_decision_tree,
                args.do_naive_bayes,
            ]
        ), "Only one classifier can be selected at a time when do_cnn do_vit is true"
    else:
        assert any(
            [
                args.do_svm,
                args.do_knn,
                args.do_random_forest,
                args.do_mlp,
                args.do_logistic_regression,
                args.do_ensemble_adaboost,
                args.do_decision_tree,
                args.do_naive_bayes,
            ]
        ), "At least one classifier should be selected"
