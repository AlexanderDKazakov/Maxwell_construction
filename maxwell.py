#!/usr/bin/env python3
import argparse
import sys
import numpy as np
from typing import Tuple, Union, List
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from dataclasses       import dataclass, field
from scipy.signal      import argrelextrema
from scipy.integrate   import quad
import random
import time

import itertools
spinner = itertools.cycle(['-', '/', '|', '\\'])

try:
    from vplotter import Plotter
    plotter_available = True
except Exception:
    print("[Maxwell Construction] Plotter is not available.")
    plotter_available = False

@dataclass
class Maxwell:
    __version__        : str                              = "0.3.1 [135]"
    internal_name      : str                              = "[Maxwell Construction]"
    verbose            : bool                             = False
    filename           : str                              = None
    region_of_interest : Union[str, Tuple[float, float]]  = "all"
    x                  : List[float]                      = None
    y                  : List[float]                      = None
    number_of_points   : int                              = -1
    iteration_limit    : int                              = 100
    tolerance          : float                            = 0.5
    return_best        : bool                             = False
    use_cols           : Tuple[int, int]                  = (0, 1)

    should_plot        : bool                             = False
    _can_calculate     : bool                             = True
    _internal_error    : int                              = 0
    _notified          : bool                             = False
    #self.xdata4fit = None
    _x                 : np.array                         = None
    _y                 : np.array                         = None
    _interpolations    : List                             = field(default_factory=list)
    _rev_interpolations: List                             = field(default_factory=list)
    #self.ydata4fit = None
    _p_left_bound      : float                            = None
    _p_left            : float                            = None
    _p_minimum         : float                            = None
    _p_maximum         : float                            = None
    _p_right           : float                            = None
    _p_right_bound     : float                            = None
    # -1 hack
    # 0 -- everything is okay # 1 -- can't | monotonic decay # 2 -- can't | right tail is not long enough ...

    _plotter           : Plotter                          = None
    _plottedL          : bool                             = False
    _plottedC          : bool                             = False
    _plottedR          : bool                             = False
    _plottedA          : bool                             = False

    def __post_init__(self):
        # Should be last
        if self.should_plot and plotter_available:
            if self.verbose: print(f"{self.internal_name} Plotting provided data..")
            self.plotter = Plotter(yname="Pressure", xname="Volumes", xlog=True,
                                   ymax= 100, # TODO: either argument or none!
                                   ymin= -20, # TODO: either argument or none!
                                   )

        self.load(x=self.x, y=self.y, filename=self.filename)

        self._xy = np.column_stack((self._x, self._y))
        # from low to high with respect to V
        self._xy = self._xy[self._xy[:,0].argsort()]

        if self.verbose:
            print(f"Loaded:")
            print(f"        x {self._x.size}")
            print(f"        y {self._y.size}")
            print(f"       xy {self._xy.size}")


        self.processing_data(_xy=self._xy)

        # Should be last
        if self.should_plot and plotter_available:
            # plot provided data
            self.plotter.plot(x=self._xy[:,0], y=self._xy[:,1], key_name="provided data")

        #maxwell_curve_xy = self.get_maxwell_curve()
        #if self.should_plot and plotter_available:
        #    self.plotter.plot(x=maxwell_curve_xy[:,0], y=maxwell_curve_xy[:,1], key_name="maxwell curve")


    @staticmethod
    def find_extrems(data, type_extremum="min/max") -> np.array:
        if type_extremum == "min": criterion = np.less
        else:                      criterion = np.greater
        try:
            out = data[argrelextrema(data[:,1], criterion, mode="clip", )]
            # if no out at all
            if out.size == 0: out = data[argrelextrema(data[:,1], criterion, mode="wrap",)]
        except Exception as e:
            print(f"Trouble: {e}")
            out = -100500
        return out

    @staticmethod
    def _check_xy(x: List[float], y: List[float]):
        x, y = np.array(x), np.array(y)
        good_values = ~np.isnan(y)
        _x, _y = x[good_values], y[good_values]
        return _x, _y

    @staticmethod
    def running_mean(x, y, N_out=101, sigma=1):
        '''
        Returns running mean as a Bell-curve weighted average at evenly spaced
        points. Does NOT wrap signal around, or pad with zeros.

        Arguments:
        x -- x values for array
        y -- y values, the values to be smoothed and re-sampled

        Keyword arguments:
        N_out -- No of elements in resampled array.
        sigma -- 'Width' of Bell-curve in units of param x .
        '''
        N_in = len(y)

        # Gaussian kernel
        x_out = np.linspace(np.min(x), np.max(x), N_out)
        x_mesh, x_out_mesh = np.meshgrid(x, x_out)
        gauss_kernel = np.exp(-np.square(x_mesh - x_out_mesh) / (2 * sigma**2))
        # Normalize kernel, such that the sum is one along axis 1
        normalization = np.tile(np.reshape(np.sum(gauss_kernel, axis=1), (N_out, 1)), (1, N_in))
        gauss_kernel_normalized = gauss_kernel / normalization
        # Perform running average as a linear operation
        y_out = gauss_kernel_normalized @ y
        return x_out, y_out

    @staticmethod
    def _missed_part(p1: Tuple[float, float], p2: Tuple[float, float], between_points):
        if between_points is not None: between = np.vstack((p1, between_points, p2))
        else: between = np.vstack((p1, p2))

        #print("Between: ", between)
        # calculate appropriate sigma
        sigma = abs(p1[0] - p2[0])/20
        #print("sigma: ", sigma)

        x_between, y_between = Maxwell.running_mean(
            x=between[:,0],
            y=between[:,1],
            N_out=30,
            sigma=sigma# 0.0001  #
        )
        xy_between = np.column_stack((x_between, y_between))
        return xy_between

    def get_inter(
        self,
        p1:Tuple[float, float],
        p2:Tuple[float, float],
        inversed: bool = False,
    ):
        x_id = 0
        y_id = 1
        idxs = (self._xy[:,x_id]>p1[x_id])&(self._xy[:,x_id]<p2[x_id])

        #if self.verbose: print(f"p1: {p1}, p2: {p2}")

        if inversed:
            x = self._xy[idxs][:,y_id]
            y = self._xy[idxs][:,x_id]
        else:
            x = self._xy[idxs][:,x_id]
            y = self._xy[idxs][:,y_id]

        #  ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’.
        return interp1d(x, y, kind='cubic')

    def fit(self,
            p1:Tuple[float, float],
            p2:Tuple[float,float],
            key_name:str="",
            ) -> np.array:
        full  = True

        try:
            inter     = self.get_inter(p1=p1, p2=p2)
            rev_inter = self.get_inter(p1=p1, p2=p2, inversed=True)
            self._interpolations.append((key_name,inter))
            self._rev_interpolations.append((key_name,rev_inter))
        except ValueError:
            # not possible to interpolate this region
            return
        x = np.linspace(p1[0], p2[0], 10000)

        while True:
            try:
                y = inter(x)
                break
            except ValueError:
                x = x[1:-1]
                full = False
        # preparing output
        xy = np.column_stack((x,y))
        if self.verbose:
            print(f"size: {xy.shape}")

        # fitting with running mean
        if not full:
            # left
            last_p_left = np.array((x[0], inter(x[0])))
            xy_add = self.fit(p1, last_p_left, key_name=key_name)
            # put in front of x
            xy = np.insert(xy, 0, xy_add, axis=0)
            if self.verbose:
                print(f"size: {xy.shape}")

            # right
            first_p_right = np.array((x[-1], inter(x[-1])))
            xy_add = self.fit(first_p_right, p2, key_name=key_name)
            # put at the end of x
            xy = np.insert(xy, -1, xy_add, axis=0)
            if self.verbose:
                print(f"size: {xy.shape}")

        #self.plotter.plot(x=xy[:,0], y=xy[:,1], key_name=key_name,)
        return xy

    # Provided user input for *x* and *y* or *filename*
    # The function would initialized internal representation of array --> self._x and self._y
    def load(self,
             x: List[float] = None,
             y: List[float] = None,
             filename: str  = None,
             ):
        # reading all data
        if x is not None and y is not None and filename is not None:
            print("Error! You provided *x* and *y* as well as *filename*. What should be used?")
            sys.exit(1)

        # x and y provided
        if x is not None and y is not None:
            if self.verbose: print(f"{self.internal_name} Checking data... [NaN...]")
            self._x, self._y = Maxwell._check_xy(x=x, y=y)
        elif filename is not None:
            if self.verbose: print(f"{self.internal_name} File opening... [{self.filename}]")
            try:
                self._x, self._y = np.loadtxt(self.filename, unpack=True, usecols=self.use_cols);
            except Exception as e:
                print(f"Error: {e}");
                sys.exit(1)
        else:
            print(f"Error! You have to provide either *x* and *y* [V vs. p] or *filename* ")
            sys.exit(1)

        # if provided the *region_of_interest* Union[str, Tuple[float, float]]
        if self.region_of_interest != "all":
            values_ids = (self._x > self.region_of_interest[0]) & (self._x < self.region_of_interest[1])
            self._x = self._x[values_ids]
            self._y = self._y[values_ids]

    @staticmethod
    def integrate(a, b, fs=["fun1", "fun2"]):
        def f(x):
            for f in fs:
                try: val = f(x)
                except: pass
            return val

        return quad(f, a=a, b=b)[0]  # return value integral between a, b

    def processing_data(self, _xy: np.array):
        #
        self.smallest_i       = np.nan
        self.maxwell_p        = np.nan
        self.smallest_diff    = 100500
        self.left_part_final  = 0
        self.right_part_final = 0
        #
        self._p_left_bound  = _xy[0]
        self._p_right_bound = _xy[-1]
        # find extremes
        self._p_maximum = maximum = Maxwell.find_extrems(_xy, "max")
        self._p_minimum = minimum = Maxwell.find_extrems(_xy, "min")

        if self.verbose:
            print(f"Maximum: {maximum}")
            print(f"Minimum: {minimum}")

        # check the number of extremes: must be one maximum and one minimum
        if maximum.shape[0] != 1 or minimum.shape[0] != 1:
            print("Warning! More than max/min was found!")
            # max
            i=0
            for maxi in maximum: print(f"Maximum[{i}]: {maxi}"); i+=1;
            # min
            i=0
            for mini in minimum: print(f"Minimum[{i}]: {mini}"); i+=1
            # if found more than 1 maximum -> take the smallest
            self._p_maximum = maximum[maximum[:,0].argmax()]
            # if found more than 1 minimum -> take the smallest
            self._p_minimum = minimum[minimum[:,0].argmin()]
            print(f"""
Taken (minimal available):
    [maximum]: {self._p_maximum}
    [minimum]: {self._p_minimum}
""")
        else:
            # take the inner point
            self._p_maximum = self._p_maximum[0]
            self._p_minimum = self._p_minimum[0]

        # check how far extremes
        # Do we really need it?
        if self._p_maximum[0] < self._p_minimum[0]:
            self.summary()
            print(f"Error! Volume corresponding to maximum [{self._p_maximum[0]}] is lower than to minimum [{self._p_minimum[0]}]!")
            return

        if self.should_plot and plotter_available:
            self.plotter.plot(x=self._p_left_bound[0],  y=self._p_left_bound[1],  key_name="Lbound", plot_line=False,)
            self.plotter.plot(x=self._p_minimum[0],     y=self._p_minimum[1],     key_name="min",    plot_line=False,)
            self.plotter.plot(x=self._p_maximum[0],     y=self._p_maximum[1],     key_name="max",    plot_line=False,)
            self.plotter.plot(x=self._p_right_bound[0], y=self._p_right_bound[1], key_name="Rbound", plot_line=False,)

        self.p1 = self._p_left_bound
        self.p2 = self._p_minimum

        # point_left_bound, p_minimum, p_maximum, point_right_bound
        inter_xy = self.fit(p1=self._p_left_bound, p2=self._p_minimum,     key_name="L")
        inter_xy = self.fit(p1=self._p_minimum,    p2=self._p_maximum,     key_name="C")
        inter_xy = self.fit(p1=self._p_maximum,    p2=self._p_right_bound, key_name="R")
        # all fit are done
        # starting from maximum and go to the minimum
        self.Vl_Vc_defined = [];
        self.Vc_Vr_defined = []
        for part, inter in self._interpolations:
            if part == "L": self.Vl_Vc_defined.append(inter)
            if part == "R": self.Vc_Vr_defined.append(inter)
            if part == "C":
                self.Vl_Vc_defined.append(inter)
                self.Vc_Vr_defined.append(inter)

        print("Working...")
        if self.number_of_points != -1: self.go_brute()
        else:                           self.go_smart()

        if self.smallest_diff > self.tolerance and self.smallest_diff != 100500 and not self.return_best:
            print(f"""
Maxwell found at step [{self.smallest_i}] some pressure [{self.maxwell_p:5.4}], but the area difference [{self.smallest_diff:4.5}] is higher than tolerance [{self.tolerance}]. Thus the Maxwell pressure will be reset.
""")
            self.maxwell_p        = np.nan
            self.smallest_diff    = 0
            self.left_part_final  = 0
            self.right_part_final = 0

        if self.should_plot and plotter_available:
            try:
                self.plotter.plot(y=self._p_left[1],   x=self._p_left[0],   key_name="L", plot_line=False,)
                self.plotter.plot(y=self._p_center[1], x=self._p_center[0], key_name="C", plot_line=False,)
                self.plotter.plot(y=self._p_right[1],  x=self._p_right[0],  key_name="R", plot_line=False,)
            except: pass
        self.summary()
        if self.should_plot and plotter_available:
            if not np.isnan(self.maxwell_p):
                self.plotter.plot(
                    y=[self.maxwell_p for i in range(10)],
                    x=np.linspace(0, 1, 10),
                    key_name_f="pressure" + str('{:.3f}'.format(self.maxwell_p)))

    def go_brute(self):
        i_growth = 0
        for i, p_c in enumerate(np.linspace(self._p_maximum[1], self._p_minimum[1], self.number_of_points)):
            if i_growth > 1000: break
            Vl, Vc, Vr = self.get_Vs(p_c=p_c)
            if np.isnan(Vl) or np.isnan(Vc) or np.isnan(Vr): continue

            left_part, right_part = self.get_parts(p_c=p_c, Vl=Vl, Vc=Vc, Vr=Vr)
            if left_part is None or right_part is None: continue

            diff = abs(right_part - left_part)
            if self.smallest_diff > diff:
                self.bookkeeping(p_c=p_c, Vl=Vl, Vc=Vc, Vr=Vr, left_part=left_part, right_part=right_part)
                self.smallest_diff = diff
                self.smallest_i    = i
            else:
                i_growth += 1
            if self.verbose:
                print(f"[i:{i:3}] p: {p_c:5.4} | diff: {diff:2.3f}")

    def go_smart(self):
        diff_prev = 100500
        diff      = 100500

        p_c = self._p_maximum[1] # initial value
        self.p_c_prev = p_c

        eta = abs(self._p_maximum[0] - self._p_minimum[0]) / 10
        grad_f = 0.1 #(self.p_c_prev-p_c)/(diff_prev - diff)
        grad_f_priv = grad_f
        i = 0
        while True:
            i += 1
            #print(f"[i:{i:3}]: p: {p_c:5.4} | eta: {eta:5.3} | grad: {grad_f:+2.3e} | diff: {diff:2.3f}")
            if i >= self.iteration_limit: break
            if self.smallest_diff < self.tolerance: break

            if i==1: p_c = self._p_maximum[1]  # initial_value
            if p_c < self._p_minimum[1] or p_c > self._p_maximum[1]:
                p_c = random.uniform(self._p_minimum[1], self._p_maximum[1],)  # initial_value

            p_c = p_c - eta * grad_f

            Vl, Vc, Vr = self.get_Vs(p_c=p_c)
            if np.isnan(Vl) or np.isnan(Vc) or np.isnan(Vr): continue

            left_part, right_part = self.get_parts(p_c=p_c, Vl=Vl, Vc=Vc, Vr=Vr)
            if left_part is None or right_part is None: continue

            diff = abs(right_part - left_part)
            if self.smallest_diff > diff:
                self.bookkeeping(p_c=p_c, Vl=Vl, Vc=Vc, Vr=Vr, left_part=left_part, right_part=right_part)
                self.smallest_diff = diff
                self.smallest_i = i

            delta_p_c     = self.p_c_prev - p_c
            delta_diff    = diff_prev - diff
            grad_f        = delta_p_c/delta_diff
            self.p_c_prev = p_c
            diff_prev     = diff
            eta = diff*1/abs(grad_f - grad_f_priv)
            if eta > 1.: eta = 1.0
            print(f"[i:{i:3}] p: {p_c:5.4} | eta: {eta:5.3} | grad: {grad_f:2.3e} | diff: {diff:2.3e}")


    def bookkeeping(self, p_c, Vl, Vc, Vr, left_part, right_part):
        self.maxwell_p        = p_c
        self._p_left          = np.array((Vl, p_c))
        self._p_center        = np.array((Vc, p_c))
        self._p_right         = np.array((Vr, p_c))
        self.left_part_final  = left_part
        self.right_part_final = right_part


    def get_Vs(self, p_c):
        Vl, Vc, Vr = np.nan, np.nan, np.nan
        for idx, (part, inter) in enumerate(self._rev_interpolations):
            #print(idx, part, inter)
            if part == "C":
                try:
                    Vc = inter(p_c)
                    #self.plotter.plot(y=p_c, key_name="C", plot_line=False, animation=True, x=Vc,)
                    #time.sleep(0.01)
                except: pass
            if part == "L":
                try:
                    Vl = inter(p_c)
                    #self.plotter.plot(y=p_c, key_name="L", plot_line=False, animation=True, x=Vl,)
                    #time.sleep(0.01)
                except: pass
            if part == "R":
                try:
                    Vr = inter(p_c)
                    #self.plotter.plot(y=p_c, key_name="R", plot_line=False, animation=True, x=Vr,)
                    #time.sleep(0.01)
                except: pass
        return Vl, Vc, Vr

    def get_parts(self, p_c, Vl, Vc, Vr):
        left_part = None
        right_part = None
        try:
            left_part  = p_c * (Vc - Vl) - self.integrate(a=Vl, b=Vc, fs=self.Vl_Vc_defined)
            right_part = self.integrate(a=Vc, b=Vr, fs=self.Vc_Vr_defined) - p_c * (Vr - Vc)
        except Exception as e:
            pass

        return left_part, right_part

    def summary(self):
        print(f"""
Summary:
    Maxwell pressure {self.maxwell_p}
    Area difference {np.nan if np.isnan(self.maxwell_p) else self.smallest_diff} | tolerance/N points [{self.tolerance if self.number_of_points == -1 else self.number_of_points}]
    Area[L] {self.left_part_final} | [R] {self.right_part_final}
""")

    def get_volumes(self):
        '''
        Returns corresponding volumes:
        Vleft -- volume of cross left
        Vmin  -- volume of minimum
        Vmax  -- volume of maximum
        Vright -- volume of cross right
        '''
        if not np.isnan(self.maxwell_p):
            return (self._p_left[0], self._p_minimum[0], self._p_maximum[0], self._p_right[0])
        else:
            # not defined
            return (np.nan, np.nan, np.nan, np.nan)

    def get_tolerance(self):
        if self.smallest_diff != 100500:
            return self.smallest_diff
        else: return np.nan

    def get_maxwell_curve(self):
        # introduce left_point and right_point in the data provided by user
        # substitute in between by maxwell_p
        maxwell_curve = np.copy(self._xy)
        # if maxwell success
        if not np.isnan(self.maxwell_p):
            # adding the left point and right
            maxwell_curve = np.insert(maxwell_curve, 0, self._p_left, axis=0)
            maxwell_curve = np.insert(maxwell_curve, 0, self._p_right, axis=0)

            maxwell_curve = maxwell_curve[maxwell_curve[:,0].argsort()]
            # substitute in between
            maxwell_curve[:,1][(
                (maxwell_curve[:,0]>=self._p_left[0]) & (maxwell_curve[:,0] <= self._p_right[0])
            )] = self.maxwell_p
            maxwell_curve = maxwell_curve[maxwell_curve[:,0].argsort()]
        return maxwell_curve


if __name__=="__main__":
    my_parser = argparse.ArgumentParser(
        description="Maxwell Construction procedure on provided data"
    )

    my_parser.add_argument('-i',  '--input',
                           action='store', type=str, required=True,
                           help="Input file path",
                           )
    my_parser.add_argument('-r',  '--region_of_interest',
                           action='store', type=str, required=False,
                           help="Region of interest in terms of volume. Ex. 0.2:10. Default: 'all'",
                           )
    my_parser.add_argument('-n',  '--number_of_points',
                           action='store', type=int, required=False,
                           help="Number of point to use when fit you data with spline. Default: '-1'.\n"
                           "All positive values refer to number of points '-1' refers to smart search.",
                           )
    my_parser.add_argument('-t',  '--tolerance',
                           action='store', type=float, required=False,
                           help="Area tolerance difference. Default: '0.5'",
                           )
    my_parser.add_argument('-l',  '--iteration_limit',
                           action='store', type=int, required=False,
                           help="Iteration limit. Default: '100'",
                           )
    my_parser.add_argument('-x',  '--x_column',
                           action='store', type=int, required=False,
                           help="Volume column in the file. Default: '0'",
                           )
    my_parser.add_argument('-y',  '--y_column',
                           action='store', type=int, required=False,
                           help="Pressure column in the file. Default: '1'",
                           )
    my_parser.add_argument('--verbose', dest='verbose', action='store_true',  help="make the Maxwell verbose")
    my_parser.add_argument('--plot', dest='plot', action='store_true',  help="plotting")
    my_parser.add_argument('--return_best', dest='return_best', action='store_true',  help="return anything what found")
    my_parser.set_defaults(region_of_interest="all")
    my_parser.set_defaults(number_of_points=-1)
    my_parser.set_defaults(x_column=0)
    my_parser.set_defaults(y_column=1)
    my_parser.set_defaults(verbose=False)
    my_parser.set_defaults(plot=False)
    my_parser.set_defaults(return_best=False)
    my_parser.set_defaults(tolerance=0.5)
    my_parser.set_defaults(iteration_limit=100)
    args = my_parser.parse_args()

    if args.region_of_interest != "all":
        region_of_interest = args.region_of_interest.split(":")
        region_of_interest = (
            float(region_of_interest[0]) if region_of_interest[0] else -np.inf,
            float(region_of_interest[1]) if region_of_interest[1] else np.inf
        )
    else:
        region_of_interest = args.region_of_interest

    #print(args)
    m = Maxwell(
        filename=args.input,
        region_of_interest=region_of_interest,
        number_of_points=args.number_of_points,
        should_plot=args.plot,
        tolerance= args.tolerance,
        iteration_limit=args.iteration_limit,
        verbose=args.verbose,
        return_best=args.return_best,
    )
    input("Press any key to continue...")
