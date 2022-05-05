# 1. Maxwell construction

- [1. Maxwell construction](#1-maxwell-construction)
  - [1.1. Usage](#11-usage)
  - [1.2. Contribution](#12-contribution)
  - [Future](#future)

A small script for creating Maxwell construction on your data with non-monotonic behaviour like on figure ([see](pic/idea.png)).
Created for internal needs. Feel free to use it and contribute (see [contribution section](#12-contribution)).

![idea](pic/idea.png)

## 1.1. Usage

```
usage: maxwell.py [-h] -i INPUT [-e ERROR_PRESSURE] [-a REGION_OF_INTEREST_VOLUME] [-o REGION_OF_INTEREST_PRESSURE] [-n NUMBER_OF_POINTS] [-g GROWTH_LIMIT] [-s SMOOTHNESS_LEVEL] [-t TOLERANCE]
                  [-l ITERATION_LIMIT] [-x X_COLUMN] [-y Y_COLUMN] [--verbose] [--plot] [--return_best] [--interpolate] [--number_inter_points NUMBER_INTER_POINTS]

Maxwell Construction

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input file path
  -e ERROR_PRESSURE, --error_pressure ERROR_PRESSURE
                        A certain value of percent uncertainty of finding pressure. Default: 0. Ex. 10. Means 10 percent of uncertainty
  -a REGION_OF_INTEREST_VOLUME, --region_of_interest_volume REGION_OF_INTEREST_VOLUME
                        Region of interest in terms of volume. Ex. 0.2:10. Default: 'all'
  -o REGION_OF_INTEREST_PRESSURE, --region_of_interest_pressure REGION_OF_INTEREST_PRESSURE
                        Region of interest in terms of pressure. Ex. 20:-5. Default: 'all'
  -n NUMBER_OF_POINTS, --number_of_points NUMBER_OF_POINTS
                        Number of point to use when fit you data with spline. Default: '-1'. All positive values refer to brute search, whereas '-1' refers to smart search.
  -g GROWTH_LIMIT, --growth_limit GROWTH_LIMIT
                        Growth delta area limit. Default: '100'
  -s SMOOTHNESS_LEVEL, --smoothness_level SMOOTHNESS_LEVEL
                        Smoothness level. Default: '0' An increase led to making the data smoother
  -t TOLERANCE, --tolerance TOLERANCE
                        Area tolerance difference. Default: '0.5'
  -l ITERATION_LIMIT, --iteration_limit ITERATION_LIMIT
                        Iteration limit. Default: '100'
  -x X_COLUMN, --x_column X_COLUMN
                        Volume column in the file. Default: '0'
  -y Y_COLUMN, --y_column Y_COLUMN
                        Pressure column in the file. Default: '1'
  --verbose             make the Maxwell verbose
  --plot                plotting (required vplotter package)
  --return_best         return the best pressure if not reached required tolerance
  --interpolate         interpolate the input data
  --number_inter_points NUMBER_INTER_POINTS
                        Number of point to use when interpolation happens with spline. Default: '1000'.

```

There are two ways to use the script:

1. Call directly on the data:
```
python3 maxwell.py -i ./my_fancy_data.dat -n 1000

```
The output will be something like this:

```
[Maxwell Construction] Working...
[Maxwell Construction] Summary:
    Maxwell pressure[101] 11.528347858011582
    Area difference 0.019975927873626453 | tolerance/N points [1000]
    Area[L] 3.810065852429214 | [R] 3.8300417803028406

Press any key to continue...

```

2. Use via python:

```python3

from maxwell import Maxwell
m = Maxwell(
    filename         = "./path/to/data/.Vpperr",  # will take first column as V, second as p | see defaults
    tolerance        = 1e-7,
    number_of_points = 10000,
    return_best      = True,
)

m.get_volumes()           # will return volumes spi/binodals 
m.get_maxwell_pressure()  # returns found maxwell pressure
m.get_maxwell_curve()     # returns maxwell curve with plateau

```

The Maxwell construction can work without [Plotter](https://github.com/AlexanderDKazakov/Plotter) instance, but typically the Plotter instance quite handy.

## 1.2. Contribution

Feel free to contribute to the project, but please create initially an issue with detailed problem and way to resolve it. 

