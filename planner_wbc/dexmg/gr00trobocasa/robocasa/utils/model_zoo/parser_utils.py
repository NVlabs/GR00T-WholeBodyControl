import argparse


def get_base_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_asset_path",
        type=str,
        help="(optional) base asset path where model files are stored. "
        "Defaults to robosuite_model_zoo/assets_private folder.",
    )
    parser.add_argument(
        "--model_folder",
        type=str,
        help="(optional) folder to store model (relative to base asset path). " "Defaults to None.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        help="(optional) name of saved mjcf model. " "Defaults to name from --model_path.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="1d scaling of input model",
    )
    parser.add_argument(
        "--show_sites",
        action="store_true",
        help="show model bounds sites",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print detailed information for debugging",
    )

    ### V-HACD settings ###
    parser.add_argument(
        "--vhacd_max_output_convex_hulls",
        type=float,
        default=32,
    )
    parser.add_argument(
        "--vhacd_voxel_resolution",
        type=float,
        default=100000,
    )
    parser.add_argument(
        "--vhacd_volume_error_percent",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--vhacd_max_hull_vert_count",
        type=float,
        default=64,
    )

    return parser
