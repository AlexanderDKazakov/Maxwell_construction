#!/usr/bin/python3
import sys
import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline

try:
    from Plotter import Plotter
    plotter_available = True
except Exception:
    print("[Maxwell Construction] Plotter is not available.")
    plotter_available = False

from scipy.signal import argrelextrema
from scipy.integrate import quad

import itertools
spinner = itertools.cycle(['-', '/', '|', '\\'])


class Maxwell:
    def __init__(self, filename=None, region_of_interest=None, number_of_points=100, x=None, y=None, tolerance=1e-3, verbose=False):
        self.internal_name = "[Maxwell Construction]"
        print(self.internal_name, "v.0.2.1 [121]")
        self.filename  = filename
        self.tolerance = tolerance
        self.verbose   = verbose

        if region_of_interest != "all": self.region_of_interest = [float(x) for x in region_of_interest.split(":")]
        else: self.region_of_interest = "all"

        self.number_of_points = int(number_of_points)
        self.x = None
        self.y = None

        self.plot = False
        self.can_calculate = True

        self.xdata4fit = None
        self.ydata4fit = None

        self.plotter = None

        self.plottedL = False
        self.plottedC = False
        self.plottedR = False
        self.plottedA = False
        self.plottedData = False

        self.load(x=x,y=y)

    @staticmethod
    def fit_region(x, y): return interp1d(x, y, kind='cubic')

    @staticmethod
    def find_extrems(data, type_extremum="min/max"):
        if type_extremum == "min":
            try: out = data[argrelextrema(data, np.less)]
            except Exception as e: print("Trouble: ", e); out = -100500
            return out
        if type_extremum == "max":
            try: out = data[argrelextrema(data, np.greater)]
            except Exception as e: print("Trouble: ", e); out = -100500
            return out

    @staticmethod
    def get_correct_extremums(data, extrems):
        correct_ext = 0; volume4extr = []
        for extrem in extrems:
            _ = data[:,1] == extrem
            cor_vol = data[:,0][_]
            volume4extr.append(cor_vol)
        try:
            correct_extr_volume = min(volume4extr)
            indx_correct_volume = data[:,0] == correct_extr_volume
            correct_ext = data[:,1][indx_correct_volume]
        except ValueError: return extrems
        return correct_ext

    def plot_data_of_interest(self):
        if self.verbose: print(self.internal_name, "Here is what you have provided... plotting...")
        if plotter_available: self.plotter = Plotter(yname="Pressure", xname="Volumes",xlog=True, ymin="auto", ymax="auto")
        # data of interest
        if not self.plottedData:
            if plotter_available: self.plotter.plot(x=self.xydata4fit[:,0], y=self.xydata4fit[:,1], key_name_f="data_interest")
            self.plottedData=True

    def check_xy(self, x,y):
        if self.verbose: print(self.internal_name, "checking data...")
        if self.verbose: print(" [Checking on] --> nan...")
        x, y = np.array(x), np.array(y)
        good_values = ~np.isnan(y)
        _x, _y = x[good_values], y[good_values]
        return _x, _y

    def load(self,x=None, y=None):
        try: len(x); array_passed = True
        except TypeError: array_passed = False

        if array_passed: self.x, self.y = self.check_xy(x=x, y=y)
        else:
            if self.filename:
                if self.verbose: print(self.internal_name, f"Opening... [{self.filename}]")
                # 2 colomns formating
                try: self.x, self.y = np.loadtxt(self.filename, unpack=True, usecols=(0,1)); loaded = True
                except Exception as e: print("Trouble:", e); loaded = False
                if not loaded:
                    # very strange format: <val>\t<val2>\n
                    self.a = _ = []
                    with open(self.filename) as infile:
                        self.lines = lines = infile.read().splitlines()
                        z = [line.split("\t") for line in lines]
                        for i,j in z:
                            if (i != "None" or j != "None") and (i != "nan" and j != "nan"):
                                try:
                                    _.append((float(i), float(j)))
                                except Exception as e:
                                    print("WARNING! ", e, "for i,j '", i, j, "'")
                        self.x, self.y = np.array(_)[:,0], np.array(_)[:,1]
            else:
                print(self.internal_name, "WARNING! You have not provided filename")
                sys.exit(0)

        if self.region_of_interest != "all":
            need_values = (self.x > self.region_of_interest[0]) & (self.x < self.region_of_interest[1])
            self.xdata4fit  = self.x[need_values]
            self.ydata4fit  = self.y[need_values]
        else:
            self.xdata4fit = self.x
            self.ydata4fit = self.y
        self.region_of_interest = [self.x[0], self.x[-1]]
        self.xydata4fit = np.array(list(zip(self.xdata4fit, self.ydata4fit)))
        if self.plot: self.plot_data_of_interest()

        tol_difference = 1.5   # TODO: now it is a HACK. Need to connect splines gap
        extrem1 = Maxwell.find_extrems(self.ydata4fit, "max")
        extrem2 = Maxwell.find_extrems(self.ydata4fit, "min")

        extrem1 = Maxwell.get_correct_extremums(self.xydata4fit, extrem1)
        extrem2 = Maxwell.get_correct_extremums(self.xydata4fit, extrem2)

        if len(extrem1) == 0 or len(extrem2)==0:
            print(self.internal_name + "For this data is impossible to create Maxwell construction...")
            self.can_calculate = False
            self.Maxwell_can_be_extended = False

        if abs(extrem1 - extrem2) < tol_difference:
            print(self.internal_name + "For this data is impossible to create Maxwell construction...")
            self.can_calculate = False
            self.Maxwell_can_be_extended = False

        if self.can_calculate:
            _, _, V1, V2 = self.extremus()  # getting extremus for splitting region
            # compare left pressure and right pressure for extremus
            left_pressure = self.ydata4fit[-1]

            self.fit(part="[R]", V1=V1, V2=V2)
            right_extremum_pressure = self.pressure_fit(V2)[0]
            if left_pressure < right_extremum_pressure:
                print(self.internal_name + "WARNING! Left pressure value is less than right extremum pressure value.\n"
                " It is hard to estimate Maxwell pressure for such system...\n"
                " Aborting...")
                self.can_calculate = False
                self.Maxwell_V1 = self.xdata4fit[-1]
                self.Maxwell_p1 = self.ydata4fit[-1]
                self.Maxwell_p_right_extremum = right_extremum_pressure
                self.Maxwell_can_be_extended = True
                print(f"Left corresponding volume {self.Maxwell_V1}, | pressure {self.Maxwell_p1}, right extremum: {right_extremum_pressure}")

        if self.can_calculate: self.calculate_areas()
        if not self.can_calculate: self.Maxwell_p = np.nan

    def fit(self, part="[A]", V1=0, V2=0):
        def get_x12(part, xydata, V1=0, V2=0):
            # data --> xydata4fit
            if part == "[A]": return (xydata[:,0][0] , xydata[:,0][-1])                   # x1 x2 # [A] -- All
            if part == "[L]": return (xydata[:,0][-1], xydata[:,0][xydata[:,0] < V1][0])
            if part == "[R]": return (xydata[:,0][xydata[:,0] > V2][-1], xydata[:,0][0])
            if part == "[C]": return (xydata[:,0][xydata[:,0] > V1][-1], xydata[:,0][xydata[:,0] < V2][0])

        # Fitting
        self.pressure_fit = Maxwell.fit_region(x=self.xydata4fit[:,0], y=self.xydata4fit[:,1])

        # add more points to fit
        x1, x2 = get_x12(part=part, xydata=self.xydata4fit, V1=V1, V2=V2)  #

        newVs = np.linspace(start=x1, stop=x2, num=self.number_of_points)  # new volumes
        pressure_data = self.pressure_fit(newVs)                           # corresponding pressure

        # Fitting reverse
        self.volume_fit = Maxwell.fit_region(x=pressure_data, y=newVs)

    def extremus(self):
        # TODO MAKE it even precise... # TODO make it as one function
        p1 = Maxwell.find_extrems(data=self.xydata4fit[:,1], type_extremum="min")
        p1 = Maxwell.get_correct_extremums(self.xydata4fit, p1)
        self.V1 = V1 =  self.xydata4fit[:,0][self.xydata4fit[:,1] == p1]

        p2 = Maxwell.find_extrems(data=self.xydata4fit[:,1], type_extremum="max")
        p2 = Maxwell.get_correct_extremums(self.xydata4fit, p2)
        self.V2 = V2 =  self.xydata4fit[:,0][self.xydata4fit[:,1] == p2]
        return p1, p2, V1, V2

    def get_correspoding_volumes(self, p_try, V1, V2):
        Vl = Vc = Vr = 0
        # Vl
        self.fit(part="[L]", V1=V1, V2=V2)
        Vl = self.volume_fit(p_try)
        # Vc
        self.fit(part="[C]", V1=V1, V2=V2)
        Vc = self.volume_fit(p_try)
        # Vr
        self.fit(part="[R]", V1=V1, V2=V2)
        Vr = self.volume_fit(p_try)
        return Vl, Vc, Vr

    def integrate(self, a, b):
        return quad(self.pressure_fit, a=a, b=b)[0]  # return value integral between a, b

    def calculate_areas(self):
        if self.verbose: print(self.internal_name, "Working...")
        _time = 0

        _, p2, V1, V2 = self.extremus()  # getting extremus for splitting region

        p_try = p2 - p2 * 0.01   # nice start   maximum pressure - 1 %
        if p_try[0] < 0: print("Initial pressure is negative... breaking..."); possible = False
        else: possible = True

        if possible:
            Vl, Vc, Vr = self.get_correspoding_volumes(p_try, V1, V2)
            print(f"Vl: {Vl}, Vc: {Vc}, Vr: {Vr}")

            if Vl < Vc < Vr:
                ok = True
                # Take all
                self.fit(part="[A]")
                left_part  = p_try * (Vc - Vl) - self.integrate(Vl, Vc)
                right_part = self.integrate(Vc, Vr) - p_try * (Vr - Vc)

                priv_area_difference = 100500
                current_difference = abs(left_part - right_part)
                while current_difference > self.tolerance:
                    if not self.verbose and _time%10==0 : print("Working " + next(spinner), end='\r')

                    Vl, Vc, Vr = self.get_correspoding_volumes(p_try, V1, V2)
                    left_part  = p_try * (Vc - Vl) - self.integrate(Vl, Vc)
                    right_part = self.integrate(Vc, Vr) - p_try * (Vr - Vc)

                    current_difference = abs(left_part - right_part)
                    if priv_area_difference < current_difference: print("Previous difference was lower than current one... breaking..."); break
                    if p_try[0] < 1: print("Pressure is less than 1 bar... breaking..."); break

                    p_old = p_try[0]
                    Vl_old = Vl
                    Vr_old = Vr
                    if self.verbose: print(self.internal_name, "area difference:", current_difference, "| p:", p_try)

                    if priv_area_difference > current_difference:
                        # TODO make it smarter..
                        if p_try < 5: p_try -= current_difference / 100
                        else: p_try -= current_difference / 10

                    priv_area_difference = current_difference

                    # animation
                    if self.plot:
                        x = np.linspace(start=self.region_of_interest[0], stop=self.region_of_interest[1], num=10)
                        y = [p_try[0] for i in range(len(x))]
                        if plotter_available: self.plotter.plot(x=x ,y=y , key_name_f="pressure" + str('{:.3f}'.format(p_try[0])), animation=True)
                    _time += 1

            else:
                print("Something strange is happend... aborting...\n"
                      "Try to reduce the range of interest...")
                ok = False
                p_old = -100500

            print("          " , end='\r')
            self.Maxwell_p = p_old
            self.Maxwell_Vl = Vl_old
            self.Maxwell_Vr = Vr_old
            if ok: left_part  = p_old * (Vc - Vl) - self.integrate(Vl, Vc)
            if ok: right_part = self.integrate(Vc, Vr) - p_old * (Vr - Vc)
            if ok: print(f"\nMaxwell pressure = {self.Maxwell_p}, | correspoding area {abs(left_part - right_part)}")
        else:
            print("We have negative initial pressure... Is there any adequate physics behind?..")
            self.Maxwell_p = 0


if __name__=="__main__":

	def help():
		print("Usage: ./Maxwell.py <filename> <region:of:interest[V]> <number_of_points:(default =100)>")
		print("  Ex.: ./Maxwell.py my_fancy_data.txt 0.2:1")
		print("  Ex.: ./Maxwell.py my_fancy_data.txt 0.2:1 100 ")
		print("  Ex.: ./Maxwell.py my_fancy_data.txt all 10000 ")

	if len(sys.argv) < 3: help(); sys.exit(0)

	filename = sys.argv[1]
	try:
		region_of_interest = sys.argv[2]
	except: region_of_interest = "all"
	try: number_of_points = sys.argv[3]
	except: number_of_points = 100
	print(f"You provided: \n filename: {filename} | region_of_interest: {region_of_interest}")

	m = Maxwell(filename=filename, region_of_interest=region_of_interest, number_of_points=number_of_points, verbose=False)
