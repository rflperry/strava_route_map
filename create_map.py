from argparse import ArgumentParser, Namespace
from src.generate_heatmaps import generate_gif, generate_heatmap

if __name__ == '__main__':
    parser = ArgumentParser(description='Generate a PNG heatmap from local Strava fit files')
                            # epilog='Report issues to https://github.com/remisalmon/Strava-local-heatmap/issues')

    parser.add_argument('--dir', default='activities/',
                        help='fit files directory')
    parser.add_argument('--filter', default='*.fit*',
                        help='fit files glob filter (default: *.fit*)')
    parser.add_argument('--year', nargs='+', default=[],
                        help='fit files year(s) filter (default: all)')
    parser.add_argument('--bounds', type=float, nargs=4, metavar='BOUND', default=[-90.0, +90.0, -180.0, +180.0],
                        help='heatmap bounding box as lat_min, lat_max, lon_min, lon_max (default: -90 +90 -180 +180)')
    parser.add_argument('--output', default='heatmap',
                        help='file name (default: heatmap)')
    parser.add_argument('--gif', action='store_true',
                        help='heatmap name (default: heatmap.png)')
    parser.add_argument('--zoom', type=int, default=-1,
                        help='heatmap zoom level 0-19 or -1 for auto (default: -1)')
    parser.add_argument('--sigma', type=int, default=2,
                        help='heatmap Gaussian kernel sigma in pixel (default: 2)')
    parser.add_argument('--orange', action='store_true',
                        help='non-grayscale background')
    parser.add_argument('--csv', action='store_true',
                        help='also save the heatmap data to a CSV file')
    parser.add_argument('--alpha', type=float, default=0.6,
                        help='alpha value for background, between 0 and 1 (default: 0.6)')
    # parser.add_argument('--crop', action='store_true',
    #                     help='crop min/max lat/lon bounds based on data.')
    parser.add_argument('--fps', type=int, default=4,
                        help='gif frames per second (default=4)')

    args = parser.parse_args()

    if args.gif:
        generate_gif(args)
    else:
        generate_heatmap(args)