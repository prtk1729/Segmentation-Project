import numpy as np
from scipy.interpolate import *
import matplotlib.pyplot as plt


def linear_interp(x_given, y_given, x_interp_given):
    # 1st construct f_linear obj. 
    # Pass the interp_domain_x to the above func_obj and get
    # y_interp
    f_linear = interp1d(x_given , y_given)
    y_interp_linear = f_linear(x_interp_given)
    return y_interp_linear

def spline_interp(x_given, y_given, x_interp_given):
    # 1st construct interp1d obj. with x_given and y_given passed 
    # Pass the interp_domain_x to the above func_obj and get
    # y_interp
    f_spline = interp1d(x_given , y_given , kind='cubic')
    y_interp_spline = f_spline(x_interp_given)
    return y_interp_spline


def plot_interp_results(x_given, y_given, x_interp_given, y_interp_linear, y_interp_spline, x_interp_domain, y_interp_domain ):
    # plot actual x, y
    plt.plot(x_given , y_given, 'o')
    # plot linear_interpolted x, y
    plt.plot(x_interp_given, y_interp_linear, '-')
    # plot spline_interpolted x, y
    plt.plot(x_interp_given, y_interp_spline, '--')

    plt.plot(x_interp_domain, y_interp_domain, ':')

    plt.legend(['data', 'linear', 'spline', 'perfect'], loc='best')

    print('mlml')
    plt.show()
    




def interpolation():
    '''* For this we have a series of datapoints
    at discrete locations
    * want to estimate datapoints in between 
    the given datapoints
    * We can linearly interpolate b/w our given function
        ** i.e b/w any 2 datapoints we can draw a striaght line
        and use the equation to find the intermediate values b/w the points
        Here, we use local information to estimate the global structure
        ** Or we can use higher order polynomials (interpolation functions)
        to approximate the curve b/w the two points. The most popluar is cubic spline
    
    Paramters: x_list and y_list
    * We create interpolation function using the interp1d()
        e.g f_linear = interp1d(x_list, y_list)
    * We can evaluate x anywhere b/w the bounds of the data using:-
            y_i = f_linear(x_i)
    '''

    # create x_given , y_given
    x_given = np.linspace(0, 10, 10)
    y_given = ( np.cos(x_given**2.0)/8.0 )

    # # create domain where the interpolated points would live in
    x_interp_domain = np.linspace(0, 10, 1000)
    y_interp_domain = np.cos(x_interp_domain**2.0)/8.0


    # interpolate to these points
    x_interp_given = np.linspace(0, 10, 100)

    y_interp_linear = linear_interp(x_given, y_given, x_interp_given)
    y_interp_spline = spline_interp(x_given , y_given,  x_interp_given)

    plot_interp_results(x_given,
                 y_given, 
                 x_interp_given, 
                 y_interp_linear,
                 y_interp_spline,
                 x_interp_domain, 
                 y_interp_domain)

interpolation()