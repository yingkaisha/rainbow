import numpy as np

import cartopy.crs as ccrs
import cartopy.geodesic as cgeo
import cartopy.feature as cfeature
from cartopy.feature import ShapelyFeature
from cartopy.io.shapereader import Reader, natural_earth

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import transforms

def precip_cmap(return_rgb=True):
    rgb_array = np.array([[0.85      , 0.85      , 0.85      , 1.        ],
                          [0.66666667, 1.        , 1.        , 1.        ],
                          [0.33333333, 0.62745098, 1.        , 1.        ],
                          [0.11372549, 0.        , 1.        , 1.        ],
                          [0.37647059, 0.81176471, 0.56862745, 1.        ],
                          [0.10196078, 0.59607843, 0.31372549, 1.        ],
                          [0.56862745, 0.81176471, 0.37647059, 1.        ],
                          [0.85098039, 0.9372549 , 0.54509804, 1.        ],
                          [1.        , 1.        , 0.4       , 1.        ],
                          [1.        , 0.8       , 0.4       , 1.        ],
                          [1.        , 0.53333333, 0.29803922, 1.        ],
                          [1.        , 0.09803922, 0.09803922, 1.        ],
                          [0.8       , 0.23921569, 0.23921569, 1.        ],
                          [0.64705882, 0.19215686, 0.19215686, 1.        ],
                          [0.55      , 0.        , 0.        , 1.        ]])
    cmap_ = mcolors.ListedColormap(rgb_array, 'precip_cmap')
    cmap_.set_over(rgb_array[-1, :])
    cmap_.set_under(rgb_array[0, :])
    if return_rgb:
        return cmap_, rgb_array
    else:
        return cmap_

def string_partial_format(fig, ax, x_start, y_start, ha, va, string_list, color_list, fontsize_list, fontweight_list):
    '''
    String partial formatting (experimental).
    
    handles = string_partial_format(fig, ax, 0., 0.5, 'left', 'bottom',
                                    string_list=['word ', 'word ', 'word'], 
                                    color_list=['r', 'g', 'b'], 
                                    fontsize_list=[12, 24, 48], 
                                    fontweight_list=['normal', 'bold', 'normal'])
    Input
    ----------
        fig: Matplotlib Figure instance. Must contain a `canvas` subclass. e.g., `fig.canvas.get_renderer()`
        ax: Matplotlib Axis instance.
        x_start: horizonal location of the text, scaled in [0, 1] 
        y_start: vertical location of the text, scale in [0, 1]
        ha: horizonal alignment of the text, expected to be either "left" or "right" ("center" may not work correctly).
        va: vertical alignment of the text
        string_list: a list substrings, each element can have a different format.
        color_list: a list of colors that matches `string_list`
        fontsize_list: a list of fontsizes that matches `string_list`
        fontweight_list: a list of fontweights that matches `string_list`
    
    Output
    ----------
        A list of Matplotlib.Text instance.
    
    * If `fig` is saved, then the `dpi` keyword must be fixed (becuase of canvas). 
      For example, if `fig=plt.figure(dpi=100)`, then `fig.savefig(dpi=100)`.
      
    '''
    L = len(string_list)
    Handles = []
    relative_loc = ax.transAxes
    renderer = fig.canvas.get_renderer()
    
    for i in range(L):
        handle_temp = ax.text(x_start, y_start, '{}'.format(string_list[i]), ha=ha, va=va,
                              color=color_list[i], fontsize=fontsize_list[i], 
                              fontweight=fontweight_list[i], transform=relative_loc)
        loc_shift = handle_temp.get_window_extent(renderer=renderer)
        relative_loc = transforms.offset_copy(handle_temp._transform, x=loc_shift.width, units='dots')
        Handles.append(handle_temp)
        
    return Handles

def ax_decorate(ax, left_flag, bottom_flag, bottom_spline=False):
    ax.grid(linestyle=':'); ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(bottom_spline)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    [j.set_linewidth(2.5) for j in ax.spines.values()]
    ax.tick_params(axis="both", which="both", bottom=False, top=False, \
               labelbottom=bottom_flag, left=False, right=False, labelleft=left_flag)
    return ax

def ax_decorate_box(ax):
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    [j.set_linewidth(2.5) for j in ax.spines.values()]
    ax.tick_params(axis="both", which="both", bottom=False, top=False, \
               labelbottom=False, left=False, right=False, labelleft=False)
    return ax

def lg_decorate(ax, loc=(0.9, 0.9)):
    LG = ax.legend(bbox_to_anchor=loc, ncol=1, prop={'size':14});
    LG.get_frame().set_facecolor('white')
    LG.get_frame().set_edgecolor('k')
    LG.get_frame().set_linewidth(0)
    return LG

def cmap_combine(cmap1, cmap2):
    colors1 = cmap1(np.linspace(0., 1, 256))
    colors2 = cmap2(np.linspace(0, 1, 256))
    colors = np.vstack((colors1, colors2))
    return mcolors.LinearSegmentedColormap.from_list('temp_cmap', colors)

# Cartopys

def get_country_geom(name_list):
    country_shapes = Reader(natural_earth(resolution='10m', category='cultural', name='admin_0_countries')).records()
    geoms = []
    for name in name_list:
        for shape_temp in country_shapes:
            if name == shape_temp.attributes['NAME_EN']:
                print('Selecting: {}'.format(name))
                geoms.append(shape_temp.geometry)
    return geoms

def _axes_to_lonlat(ax, coords):
    """(lon, lat) from axes coordinates."""
    display = ax.transAxes.transform(coords)
    data = ax.transData.inverted().transform(display)
    lonlat = ccrs.PlateCarree().transform_point(*data, ax.projection)

    return lonlat


def _upper_bound(start, direction, distance, dist_func):
    """A point farther than distance from start, in the given direction.

    It doesn't matter which coordinate system start is given in, as long
    as dist_func takes points in that coordinate system.

    Args:
        start:     Starting point for the line.
        direction  Nonzero (2, 1)-shaped array, a direction vector.
        distance:  Positive distance to go past.
        dist_func: A two-argument function which returns distance.

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    if distance <= 0:
        raise ValueError(f"Minimum distance is not positive: {distance}")

    if np.linalg.norm(direction) == 0:
        raise ValueError("Direction vector must not be zero.")

    # Exponential search until the distance between start and end is
    # greater than the given limit.
    length = 0.1
    end = start + length * direction

    while dist_func(start, end) < distance:
        length *= 2
        end = start + length * direction

    return end


def _distance_along_line(start, end, distance, dist_func, tol):
    """Point at a distance from start on the segment  from start to end.

    It doesn't matter which coordinate system start is given in, as long
    as dist_func takes points in that coordinate system.

    Args:
        start:     Starting point for the line.
        end:       Outer bound on point's location.
        distance:  Positive distance to travel.
        dist_func: Two-argument function which returns distance.
        tol:       Relative error in distance to allow.

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    initial_distance = dist_func(start, end)
    if initial_distance < distance:
        raise ValueError(f"End is closer to start ({initial_distance}) than "
                         f"given distance ({distance}).")

    if tol <= 0:
        raise ValueError(f"Tolerance is not positive: {tol}")

    # Binary search for a point at the given distance.
    left = start
    right = end

    while not np.isclose(dist_func(start, right), distance, rtol=tol):
        midpoint = (left + right) / 2

        # If midpoint is too close, search in second half.
        if dist_func(start, midpoint) < distance:
            left = midpoint
        # Otherwise the midpoint is too far, so search in first half.
        else:
            right = midpoint

    return right


def _point_along_line(ax, start, distance, angle=0, tol=0.01):
    """Point at a given distance from start at a given angle.

    Args:
        ax:       CartoPy axes.
        start:    Starting point for the line in axes coordinates.
        distance: Positive physical distance to travel.
        angle:    Anti-clockwise angle for the bar, in radians. Default: 0
        tol:      Relative error in distance to allow. Default: 0.01

    Returns:
        Coordinates of a point (a (2, 1)-shaped NumPy array).
    """
    # Direction vector of the line in axes coordinates.
    direction = np.array([np.cos(angle), np.sin(angle)])

    geodesic = cgeo.Geodesic()

    # Physical distance between points.
    def dist_func(a_axes, b_axes):
        a_phys = _axes_to_lonlat(ax, a_axes)
        b_phys = _axes_to_lonlat(ax, b_axes)

        # Geodesic().inverse returns a NumPy MemoryView like [[distance,
        # start azimuth, end azimuth]].
        return geodesic.inverse(a_phys, b_phys).base[0, 0]

    end = _upper_bound(start, direction, distance, dist_func)

    return _distance_along_line(start, end, distance, dist_func, tol)


def scale_bar(ax, location, length, metres_per_unit=1000, unit_name='km',
              tol=0.01, angle=0, color='black', linewidth=3, text_offset=0.005,
              ha='center', va='bottom', plot_kwargs=None, text_kwargs=None,
              **kwargs):
    """Add a scale bar to CartoPy axes.

    For angles between 0 and 90 the text and line may be plotted at
    slightly different angles for unknown reasons. To work around this,
    override the 'rotation' keyword argument with text_kwargs.

    Args:
        ax:              CartoPy axes.
        location:        Position of left-side of bar in axes coordinates.
        length:          Geodesic length of the scale bar.
        metres_per_unit: Number of metres in the given unit. Default: 1000
        unit_name:       Name of the given unit. Default: 'km'
        tol:             Allowed relative error in length of bar. Default: 0.01
        angle:           Anti-clockwise rotation of the bar.
        color:           Color of the bar and text. Default: 'black'
        linewidth:       Same argument as for plot.
        text_offset:     Perpendicular offset for text in axes coordinates.
                         Default: 0.005
        ha:              Horizontal alignment. Default: 'center'
        va:              Vertical alignment. Default: 'bottom'
        **plot_kwargs:   Keyword arguments for plot, overridden by **kwargs.
        **text_kwargs:   Keyword arguments for text, overridden by **kwargs.
        **kwargs:        Keyword arguments for both plot and text.
    """
    # Setup kwargs, update plot_kwargs and text_kwargs.
    if plot_kwargs is None:
        plot_kwargs = {}
    if text_kwargs is None:
        text_kwargs = {}

    plot_kwargs = {'linewidth': linewidth, 'color': color, **plot_kwargs,
                   **kwargs}
    text_kwargs = {'ha': ha, 'va': va, 'rotation': angle, 'color': color,
                   **text_kwargs, **kwargs}

    # Convert all units and types.
    location = np.asarray(location)  # For vector addition.
    length_metres = length * metres_per_unit
    angle_rad = angle * np.pi / 180

    # End-point of bar.
    end = _point_along_line(ax, location, length_metres, angle=angle_rad,
                            tol=tol)

    # Coordinates are currently in axes coordinates, so use transAxes to
    # put into data coordinates. *zip(a, b) produces a list of x-coords,
    # then a list of y-coords.
    ax.plot(*zip(location, end), transform=ax.transAxes, **plot_kwargs)

    # Push text away from bar in the perpendicular direction.
    midpoint = (location + end) / 2
    offset = text_offset * np.array([-np.sin(angle_rad), np.cos(angle_rad)])
    text_location = midpoint + offset

    # 'rotation' keyword argument is in text_kwargs.
    ax.text(*text_location, f"{length} {unit_name}", rotation_mode='anchor',
            transform=ax.transAxes, **text_kwargs)
    
#https://unidata.github.io/python-gallery/examples/Precipitation_Map.html


def xcolor(key):
    xcolor = {
    "maroon":"#800000", "dark red":"#8B0000", "brown":"#A52A2A", "firebrick":"#B22222", "crimson":"#DC143C", "red":"#FF0000",
    "tomato":"#FF6347", "coral":"#FF7F50", "indian red":"#CD5C5C", "light coral":"#F08080", "dark salmon":"#E9967A", "salmon":"#FA8072",
    "light salmon":"#FFA07A", "orange red":"#FF4500", "dark orange":"#FF8C00", "orange":"#FFA500", "gold":"#FFD700", "dark golden rod":"#B8860B",
    "golden rod":"#DAA520", "pale golden rod":"#EEE8AA", "dark khaki":"#BDB76B", "khaki":"#F0E68C", "olive":"#808000", "yellow":"#FFFF00",
    "yellow green":"#9ACD32", "dark olive green":"#556B2F", "olive drab":"#6B8E23", "lawn green":"#7CFC00", "chart reuse":"#7FFF00", "green yellow":"#ADFF2F",
    "dark green":"#006400", "green":"#008000", "forest green":"#228B22", "lime":"#00FF00", "lime green":"#32CD32", "light green":"#90EE90",
    "pale green":"#98FB98", "dark sea green":"#8FBC8F", "medium spring green":"#00FA9A", "spring green":"#00FF7F", "sea green":"#2E8B57", "medium aqua marine":"#66CDAA",
    "medium sea green":"#3CB371", "light sea green":"#20B2AA", "dark slate gray":"#2F4F4F", "teal":"#008080", "dark cyan":"#008B8B", "aqua":"#00FFFF",
    "cyan":"#00FFFF", "light cyan":"#E0FFFF", "dark turquoise":"#00CED1", "turquoise":"#40E0D0", "medium turquoise":"#48D1CC", "pale turquoise":"#AFEEEE",
    "aqua marine":"#7FFFD4", "powder blue":"#B0E0E6", "cadet blue":"#5F9EA0", "steel blue":"#4682B4", "corn flower blue":"#6495ED", "deep sky blue":"#00BFFF",
    "dodger blue":"#1E90FF", "light blue":"#ADD8E6", "sky blue":"#87CEEB", "light sky blue":"#87CEFA", "midnight blue":"#191970",
    "navy":"#000080", "dark blue":"#00008B", "medium blue":"#0000CD", "blue":"#0000FF", "royal blue":"#4169E1", "blue violet":"#8A2BE2",
    "indigo":"#4B0082", "dark slate blue":"#483D8B", "slate blue":"#6A5ACD", "medium slate blue":"#7B68EE", "medium purple":"#9370DB", "dark magenta":"#8B008B",
    "dark violet":"#9400D3", "dark orchid":"#9932CC", "medium orchid":"#BA55D3", "purple":"#800080", "thistle":"#D8BFD8", "plum":"#DDA0DD",
    "violet":"#EE82EE", "magenta":"#FF00FF", "orchid":"#DA70D6", "medium violet red":"#C71585", "pale violet red":"#DB7093", "deep pink":"#FF1493",
    "hot pink":"#FF69B4","light pink":"#FFB6C1","pink":"#FFC0CB","antique white":"#FAEBD7","beige":"#F5F5DC","bisque":"#FFE4C4",
    "blanched almond":"#FFEBCD","wheat":"#F5DEB3","corn silk":"#FFF8DC","lemon chiffon":"#FFFACD","light golden rod yellow":"#FAFAD2","light yellow":"#FFFFE0",
    "saddle brown":"#8B4513","sienna":"#A0522D","chocolate":"#D2691E","peru":"#CD853F","sandy brown":"#F4A460","burly wood":"#DEB887",
    "tan":"#D2B48C","rosy brown":"#BC8F8F","moccasin":"#FFE4B5","navajo white":"#FFDEAD","peach puff":"#FFDAB9","misty rose":"#FFE4E1",
    "lavender blush":"#FFF0F5","linen":"#FAF0E6","old lace":"#FDF5E6","papaya whip":"#FFEFD5","sea shell":"#FFF5EE","mint cream":"#F5FFFA",
    "slate gray":"#708090","light slate gray":"#778899", "light steel blue":"#B0C4DE","lavender":"#E6E6FA","floral white":"#FFFAF0","alice blue":"#F0F8FF",
    "ghost white":"#F8F8FF","honeydew":"#F0FFF0","ivory":"#FFFFF0","azure":"#F0FFFF","snow":"#FFFAFA","black":"#000000",
    "dim gray":"#696969","gray":"#808080","dark gray":"#A9A9A9","silver":"#C0C0C0","light gray":"#D3D3D3","gainsboro":"#DCDCDC",
    "white smoke":"#F5F5F5","white":"#FFFFFF"}
    return xcolor[key]
	



