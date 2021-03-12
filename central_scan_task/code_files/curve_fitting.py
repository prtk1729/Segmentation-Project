import numpy as np
from scipy.interpolate import *
import matplotlib.pyplot as plt


def plot_curves(x_given , y_given , x_poly_domain, y_domain):
    plt.plot(x_given , y_given, 'o')
    plt.plot(x_poly_domain , y_domain, '-')
    plt.legend(['actual_data', 'poly_fit_curve'], loc='best')
    # plt.ylim(len(y_given), 0)
    plt.show()



def poly_fit(x_given , y_given, pOrder, x_poly_domain):
    '''
    Functionality:
        Step 1:
            i/p: x_list, y_list, poly_order we want to fit it into 
                     p = np.polyfit(x, y, pOrder) 
                     which return p (a list of coefficient for the 
                     polynomial of that particular order)
                     where p[0] corresponds to highest power of coefficients
                     i.e 2x**2 + 3x + 5, p[0]=2, p[1]=3 , p[2]=5
        Step 2:
            Using the poly_coeff.
        np.polyval(p, x)
    '''

    p_coeff_list= np.polyfit(x_given , y_given , pOrder)
    y_poly = np.polyval(p_coeff_list, x_poly_domain)
    return y_poly, p_coeff_list


def general_curve_fitting():
    '''
    '''
    pass



def curve_fitting(x_given, y_given, x_poly_domain_lb, x_poly_domain_ub, type, plot=True, pOrder=3):
    '''
    Functionality:
        * Curve-Fitting can be used for interpolation
        * But more generally, used for approximating a set of data-points
            with some approximate function
        * e.g : Suppose we  are given data and we want to fit a specific func
            to that data ==> signal curve-fitting
    type=='poly_fit' ==> Do Polyfit
    type=='general_curve_fits' ==> 


    '''
    x_given = x_given
    y_given = y_given

    # change x_poly_domain_lb, x_poly_domain_lb, in OCT-Scan case
    x_poly_domain_lb, x_poly_domain_ub = x_poly_domain_lb, x_poly_domain_ub
    x_poly_domain = np.linspace(x_poly_domain_lb, x_poly_domain_ub, 1000) # x_data for plotting the poly_fit

    y_poly, p_coeff_list = poly_fit(x_given=x_given , 
                                    y_given=y_given, 
                                    pOrder=pOrder, 
                                    x_poly_domain=x_poly_domain)
    
    if plot == True:
        plot_curves(x_given , y_given, x_poly_domain, y_domain)

    return y_poly, p_coeff_list


    # general_curve_fit(x_given , y_given, x_poly_domain)



# x_given = [0. ,1. ,2. ,3. ,4. ,5. ]
# y_given = [0., 0.8, 0.9, 0.1, -0.8, -1.0]
# x_poly_domain_lb, x_poly_domain_ub = -2.0, 6.0
# curve_fitting(x_given, y_given, x_poly_domain_lb, x_poly_domain_ub, type='poly_fit', plot=True, pOrder=3)