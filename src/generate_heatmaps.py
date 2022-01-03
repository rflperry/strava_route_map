"""
Tools for generating heatmaps from latitude and longitude data.

MIT License

Original work Copyright (c) 2018 Remi Salmon
Modified work Copyright 2021 Ronan Perry

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# imports
import os
import glob
import time

import numpy as np
import matplotlib.pyplot as plt

from urllib.error import URLError
from urllib.request import Request, urlopen
from .parse_fit import get_dataframes

# globals
HEATMAP_MAX_SIZE = (2160, 3840) # maximum heatmap size in pixel
HEATMAP_MAX_SIZE = (2160 // 2, 3840 // 2) # (2160, 3840) # maximum heatmap size in pixel
HEATMAP_MARGIN_SIZE = 32 # margin around heatmap trackpoints in pixel

PLT_COLORMAP = 'hot' # matplotlib color map

OSM_TILE_SERVER = 'https://maps.wikimedia.org/osm-intl/{}/{}/{}.png' # OSM tile url from https://wiki.openstreetmap.org/wiki/Tile_servers
OSM_TILE_SIZE = 256 # OSM tile size in pixel
OSM_MAX_ZOOM = 19 # OSM maximum zoom level
OSM_MAX_TILE_COUNT = 100 # maximum number of tiles to download
METERS_PER_TRACKPOINT = 5.0 # Running approximation. Cycling ~= 5.0

# functions
def deg2xy(lat_deg, lon_deg, zoom):
    """Returns OSM coordinates (x,y) from (lat,lon) in degree"""

    # from https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
    lat_rad = np.radians(lat_deg)
    n = 2.0**zoom
    x = (lon_deg+180.0)/360.0*n
    y = (1.0-np.arcsinh(np.tan(lat_rad))/np.pi)/2.0*n

    return x, y

def xy2deg(x, y, zoom):
    """Returns (lat, lon) in degree from OSM coordinates (x,y)"""

    # from https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
    n = 2.0**zoom
    lon_deg = x/n*360.0-180.0
    lat_rad = np.arctan(np.sinh(np.pi*(1.0-2.0*y/n)))
    lat_deg = np.degrees(lat_rad)

    return lat_deg, lon_deg

def gaussian_filter(image, sigma):
    """Returns image filtered with a gaussian function of variance sigma**2"""

    i, j = np.meshgrid(np.arange(image.shape[0]),
                       np.arange(image.shape[1]),
                       indexing='ij')

    mu = (int(image.shape[0]/2.0),
          int(image.shape[1]/2.0))

    gaussian = 1.0/(2.0*np.pi*sigma*sigma)*np.exp(-0.5*(((i-mu[0])/sigma)**2+\
                                                        ((j-mu[1])/sigma)**2))

    gaussian = np.roll(gaussian, (-mu[0], -mu[1]), axis=(0, 1))

    image_fft = np.fft.rfft2(image)
    gaussian_fft = np.fft.rfft2(gaussian)

    image = np.fft.irfft2(image_fft*gaussian_fft)

    return image

def download_tile(tile_url, tile_file):
    """Download tile from url to file, wait 0.1s and return True (False) if (not) successful"""

    request = Request(tile_url, headers={'User-Agent':'Mozilla/5.0'})

    try:
        with urlopen(request) as response:
            data = response.read()

    except URLError:
        return False

    with open(tile_file, 'wb') as file:
        file.write(data)

    time.sleep(0.1)

    return True

def add_text(img, text, coords=(0.02, 0.95), fontsize=0.03, fill=(0, 0, 0)):
    """
    Adds text to a provided image in numpy.ndarray format.
    Coords are in relative position
    """
    from PIL import Image, ImageDraw, ImageFont
    img *= 255
    img = Image.fromarray(img.astype(np.uint8))
    width, height = img.size
    # Call draw Method to add 2D graphics in an image
    I1 = ImageDraw.Draw(img)

    # Custom font style and font size
    font = ImageFont.truetype('fonts/FreeMonoBold.ttf', int(fontsize*height))

    # Add Text to an image
    coords = (
        int(coords[0] * width),
        int(coords[1] * height)
    )
    I1.text(coords, text, font=font, fill=fill)

    return np.array(img)#.astype(np.float)

def get_lat_lon(args):
    # read fit trackpoints
    fit_files = glob.glob('{}/{}'.format(args.dir,
                                         args.filter))
    fit_files = sorted(fit_files)

    if not fit_files:
        exit('ERROR no data matching {}/{}'.format(args.dir,
                                                   args.filter))

    lat_lon_list = []
    dates = []
    lat_bound_min, lat_bound_max, lon_bound_min, lon_bound_max = args.bounds

    for fit_file in fit_files:
        print('Reading {}'.format(os.path.basename(fit_file)))

        _, points_df = get_dataframes(fit_file)

        lat_lon = points_df[['latitude', 'longitude']].to_numpy()
        # crop to bounding box
        lat_lon = lat_lon[np.logical_and(lat_lon[:, 0] > lat_bound_min,
                                                lat_lon[:, 0] < lat_bound_max), :]
        lat_lon = lat_lon[np.logical_and(lat_lon[:, 1] > lon_bound_min,
                                                lat_lon[:, 1] < lon_bound_max), :]

        if lat_lon.shape[0] > 0:
            dates.append(points_df['timestamp'].loc[0].strftime('%b %d, %Y'))
            lat_lon_list.append(lat_lon)

    lat_lon_data = np.vstack(lat_lon_list)
    print(f'Processing {lat_lon_data.shape[0]} coordinates.')

    if lat_lon_data.size == 0:
        exit('ERROR no data matching {}/{}{}'.format(args.dir,
                                                     args.filter,
                                                     ' with year {}'.format(' '.join(args.year)) if args.year else ''))

    if lat_lon_data.size == 0:
        exit('ERROR no data matching {}/{} with bounds {}'.format(args.dir, args.filter, args.bounds))

    print('Read {} trackpoints'.format(lat_lon_data.shape[0]))

    # find tiles coordinates
    # if args.crop:
    lat_min, lon_min = np.min(lat_lon_data, axis=0)
    lat_max, lon_max = np.max(lat_lon_data, axis=0)
    # else:
        # lat_min, lon_min = lat_bound_min, lon_bound_min
        # lat_max, lon_max = lat_bound_max, lon_bound_max

    bounds = [lat_min, lat_max, lon_min, lon_max]

    return lat_lon_list, dates, bounds

def get_background_map(bounds, args):
    lat_min, lat_max, lon_min, lon_max = bounds

    if args.zoom > -1:
        zoom = min(args.zoom, OSM_MAX_ZOOM)

        x_tile_min, y_tile_max = map(int, deg2xy(lat_min, lon_min, zoom))
        x_tile_max, y_tile_min = map(int, deg2xy(lat_max, lon_max, zoom))

    else:
        zoom = OSM_MAX_ZOOM

        while True:
            x_tile_min, y_tile_max = map(int, deg2xy(lat_min, lon_min, zoom))
            x_tile_max, y_tile_min = map(int, deg2xy(lat_max, lon_max, zoom))

            if ((x_tile_max-x_tile_min+1)*OSM_TILE_SIZE <= HEATMAP_MAX_SIZE[0] and
                (y_tile_max-y_tile_min+1)*OSM_TILE_SIZE <= HEATMAP_MAX_SIZE[1]):
                break

            zoom -= 1

        print('Auto zoom = {}'.format(zoom))

    tile_count = (x_tile_max-x_tile_min+1)*(y_tile_max-y_tile_min+1)

    if tile_count > OSM_MAX_TILE_COUNT:
        exit('ERROR zoom value too high, too many tiles to download')

    # download tiles
    os.makedirs('tiles', exist_ok=True)

    supertile = np.zeros(((y_tile_max-y_tile_min+1)*OSM_TILE_SIZE,
                          (x_tile_max-x_tile_min+1)*OSM_TILE_SIZE, 3))

    n = 0
    for x in range(x_tile_min, x_tile_max+1):
        for y in range(y_tile_min, y_tile_max+1):
            n += 1

            tile_file = 'tiles/tile_{}_{}_{}.png'.format(zoom, x, y)

            if not glob.glob(tile_file):
                print('downloading tile {}/{}'.format(n, tile_count))

                tile_url = OSM_TILE_SERVER.format(zoom, x, y)

                if not download_tile(tile_url, tile_file):
                    print('ERROR downloading tile {} failed, using blank tile'.format(tile_url))

                    tile = np.ones((OSM_TILE_SIZE,
                                    OSM_TILE_SIZE, 3))

                    plt.imsave(tile_file, tile)

            tile = plt.imread(tile_file)

            i = y-y_tile_min
            j = x-x_tile_min

            supertile[i*OSM_TILE_SIZE:(i+1)*OSM_TILE_SIZE,
                      j*OSM_TILE_SIZE:(j+1)*OSM_TILE_SIZE, :] = tile[:, :, :3]

    if not args.orange:
        supertile = np.sum(supertile*[0.2126, 0.7152, 0.0722], axis=2) # to grayscale
        supertile = 1.0-supertile # invert colors
        supertile = np.dstack((supertile, supertile, supertile)) # to rgb

    return supertile, zoom, x_tile_min, y_tile_min

def get_trackpoints(lat_lon_list, shape, ij_data, args):
    trackpoint_list = np.zeros((len(lat_lon_list), *shape))

    # fill trackpoints
    sigma_pixel = args.sigma #if not args.orange else 1

    # Count number of routes through pixels, with sigma radius
    for idx, ij in enumerate(ij_data):
        for i, j in ij:
            trackpoint_list[idx, i-sigma_pixel:i+sigma_pixel, j-sigma_pixel:j+sigma_pixel] += 1.0

    return trackpoint_list

def smooth_trackpoints(data, m, data_hist, args):
    sigma_pixel = args.sigma #if not args.orange else 1

    data[data > m] = m

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j] = m*data_hist[int(data[i, j])] # histogram equalization

    data = gaussian_filter(data, float(sigma_pixel)) # kernel density estimation with normal kernel

    return data

def create_heatmap(supertile, data, ij_data, args):

    # colorize
    if not args.orange:
        cmap = plt.get_cmap(PLT_COLORMAP)

        data_color = cmap(data)
        data_color[data_color == cmap(0.0)] = 0.0 # remove background color

        for c in range(3):
            supertile[:, :, c] = (1.0-data_color[:, :, c])*supertile[:, :, c]+data_color[:, :, c]

    else:
        cmap = plt.get_cmap(PLT_COLORMAP)

        data_color = cmap(data)
        data_color[data_color == cmap(0.0)] = 0 # remove background color

        for c in range(3):
            supertile[:, :, c] = (1.0-data) * args.alpha*supertile[:, :, c] + data * data_color[:, :, c]

        # color = np.array([255, 82, 0], dtype=float)/255 # orange

        # for c in range(3):
        #     supertile[:, :, c] = np.minimum(supertile[:, :, c]+gaussian_filter(data, 1.0), 1.0) # white

        # data = gaussian_filter(data, 0.5) # original given sigma 0.5
        # data = (data-data.min())/(data.max()-data.min())

        # for c in range(3):
        #     supertile[:, :, c] = (1.0-data)*supertile[:, :, c]+data*color[c]

    # crop image
    # if args.crop:
    i_min, j_min = np.min(ij_data, axis=0)
    i_max, j_max = np.max(ij_data, axis=0)

    supertile = supertile[max(i_min-HEATMAP_MARGIN_SIZE, 0):min(i_max+HEATMAP_MARGIN_SIZE, supertile.shape[0]),
                        max(j_min-HEATMAP_MARGIN_SIZE, 0):min(j_max+HEATMAP_MARGIN_SIZE, supertile.shape[1])]

    return supertile

def generate_heatmap(args):
    # Get latitude and longitude data
    lat_lon_list, dates, bounds = get_lat_lon(args)
    lat_lon_data = np.vstack(lat_lon_list)

    # Generate the background map at the zoom level
    supertile, zoom, x_tile_min, y_tile_min = get_background_map(bounds, args)

    # Create the trackpoint data map
    shape = supertile.shape[:2]
    xy_data = [np.array(deg2xy(lat_lon[:, 0], lat_lon[:, 1], zoom)).T for lat_lon in lat_lon_list]
    xy_data = [np.round((xy-[x_tile_min, y_tile_min])*OSM_TILE_SIZE) for xy in xy_data]
    ij_data = [np.flip(xy.astype(int), axis=1) for xy in xy_data] # to supertile coordinates

    trackpoint_list = get_trackpoints(lat_lon_list, shape, ij_data, args)

    res_pixel = 156543.03*np.cos(np.radians(np.mean(np.vstack(lat_lon_list)[:, 0])))/(2.0**zoom) # from https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
    m = np.round((1.0/METERS_PER_TRACKPOINT)*res_pixel*len(trackpoint_list))
    data = np.sum(trackpoint_list, axis=0)
    # threshold to max accumulation of trackpoint
    data[data > m] = m
    # equalize histogram and compute kernel density estimation
    data_hist, _ = np.histogram(data, bins=int(m+1))
    data_hist = np.cumsum(data_hist)/data.size # normalized cumulated histogram
    data = smooth_trackpoints(data, m, data_hist, args)

    data = smooth_trackpoints(np.sum(trackpoint_list, axis=0), m, data_hist, args)
    data = (data-data.min())/(data.max()-data.min()) # normalize to [0,1]

    img = create_heatmap(supertile, data, np.vstack(ij_data), args)

    img = add_text(img, f'{dates[0]} - {dates[-1]}')

    # save image
    plt.imsave('{}.png'.format(os.path.splitext(args.output)[0]), img)

    print('Saved {}.png'.format(os.path.splitext(args.output)[0]))

    # save csv
    if args.csv and not args.orange:
        csv_file = '{}.csv'.format(os.path.splitext(args.output)[0])

        with open(csv_file, 'w') as file:
            file.write('latitude,longitude,intensity\n')

            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    if data[i, j] > 0.1:
                        x = x_tile_min+j/OSM_TILE_SIZE
                        y = y_tile_min+i/OSM_TILE_SIZE

                        lat, lon = xy2deg(x, y, zoom)

                        file.write('{},{},{}\n'.format(lat, lon, data[i,j]))

        print('Saved {}'.format(csv_file))

    return

def generate_gif(args):
    import imageio
    # Get latitude and longitude data
    lat_lon_list, dates, bounds = get_lat_lon(args)
    lat_lon_data = np.vstack(lat_lon_list)

    # Generate the background map at the zoom level
    supertile, zoom, x_tile_min, y_tile_min = get_background_map(bounds, args)

    # Create the trackpoint data map
    shape = supertile.shape[:2]
    xy_data = [np.array(deg2xy(lat_lon[:, 0], lat_lon[:, 1], zoom)).T for lat_lon in lat_lon_list]
    xy_data = [np.round((xy-[x_tile_min, y_tile_min])*OSM_TILE_SIZE) for xy in xy_data]
    ij_data = [np.flip(xy.astype(int), axis=1) for xy in xy_data] # to supertile coordinates

    trackpoint_list = get_trackpoints(lat_lon_list, shape, ij_data, args)

    res_pixel = 156543.03*np.cos(np.radians(np.mean(np.vstack(lat_lon_list)[:, 0])))/(2.0**zoom) # from https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
    m = np.round((1.0/METERS_PER_TRACKPOINT)*res_pixel*len(trackpoint_list))
    data = np.sum(trackpoint_list, axis=0)
    # threshold to max accumulation of trackpoint
    data[data > m] = m
    # equalize histogram and compute kernel density estimation
    data_hist, _ = np.histogram(data, bins=int(m+1))
    data_hist = np.cumsum(data_hist)/data.size # normalized cumulated histogram
    data_final = smooth_trackpoints(data, m, data_hist, args)

    heatmaps = [create_heatmap(np.copy(supertile), np.zeros(data.shape), np.vstack(ij_data), args)]

    acts_per_sec = len(trackpoint_list) // (args.max_time * args.fps) + 1
    print(f'Creating gif at {args.fps} frames per second, {acts_per_sec} activities per frame')
    for i in range(0, len(trackpoint_list), acts_per_sec):
        # trackpoint max accumulation per pixel = 1/5 (trackpoint/meter) * res_pixel (meter/pixel) * activities
        # (Strava records trackpoints every 5 meters in average for cycling activites)
        # m = np.round((1.0/METERS_PER_TRACKPOINT)*res_pixel*(i+1))

        data = smooth_trackpoints(np.sum(trackpoint_list[:i+1], axis=0), m, data_hist, args)
        data = (data-data.min())/(data_final.max()-data_final.min()) # normalize to [0,1]

        img = create_heatmap(np.copy(supertile), data, np.vstack(ij_data), args)
        img = add_text(img, f'{dates[0]} - {dates[i]}')
        heatmaps.append(img)

    # save gif
    fps = max(args.fps, len(trackpoint_list // 15))
    imageio.mimsave('{}.gif'.format(os.path.splitext(args.output)[0]), heatmaps, fps=args.fps)
    print('Saved {}.gif'.format(os.path.splitext(args.output)[0]))

    # save image
    plt.imsave('{}.png'.format(os.path.splitext(args.output)[0]), heatmaps[-1])
    print('Saved {}.png'.format(os.path.splitext(args.output)[0]))

    return
