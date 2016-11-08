#!/Users/wengang/Applications/anaconda/bin python

# Import all necessary modules for the data read + postprocessing
#%matplotlib inline
from sys import argv, exit
import scipy.io as sio
from datetime import datetime, timedelta
import dateutil.parser
import argparse


from ecmwfapi import ECMWFDataServer
import numpy as np
import pygrib

from mpl_toolkits.basemap import Basemap
from matplotlib import animation, rc
import matplotlib.pyplot as plt

# Define the parameters for the estimations
datasets = {
    'sea_surface_temp': {
        'name': 'Sea surface temperature',
        'param': 'sst',
        'stream': 'oper'
    },
    'mean_wave_period': {
        'name': 'Mean wave period',
        'param': 'mwp',
        'stream': 'wave'
    },
    'mean_wave_dir': {
        'name': 'Mean wave direction',
        'param': 'mwd',
        'stream': 'wave'
    },
    'u_wind': {
        'name': '10 metre U wind component',
        'param': '10u',
        'stream': 'oper'
    },
    'v_wind': {
        'name': '10 metre V wind component',
        'param': '10v',
        'stream': 'oper'
    },
    'significant_wave_height': {
        'name': 'Significant height of combined wind waves and swell',
        'param': 'swh',
        'stream': 'wave'
    }
}

def datenum2datetime(matlab_datenum):
    """Function for converting Matlab's datenum to Python's datetime"""
    return (datetime.fromordinal(int(matlab_datenum)) +
            timedelta(days=matlab_datenum % 1) - timedelta(days=366))


def datetime2datenum(python_datetime):
    """
    Function: datetime2datenum(python_datetime)
    	Input: 
    		python_datetime = e.g. ['2010-01-01 01:30:00']

    	Output:
    		It will give  number of UTC corresponding to the abosulte zero time

    Function for converting Python's datetime to Matlab's datenum

    """
    if isinstance(python_datetime, str):
        python_datetime = dateutil.parser.parse(python_datetime)

    return (python_datetime.toordinal() + 366 +
            ((((python_datetime.microsecond / 1.0e6 +
                python_datetime.second) / 60. +
               python_datetime.minute) / 60.) +
             python_datetime.hour) / 24.)

# Read the data and illustrate the results of the data
def parsegribfile(filename):
	rc('animation', html='html5')

	grbs = pygrib.open(filename)  #grbs = pygrib.open("oper.grib")

	parameter = datasets['v_wind']['name']
	grb_messages = [grb for grb in grbs if grb['name']==parameter]

	# Read the data into np array
	grb = grbs.select(name=parameter)[0]
	data = grb.values  
	lats,lons = grb.latlons()

	grbs.close()

	fig = plt.figure(figsize=(8,6))
	txt = plt.title('', fontsize=8)

	global_map = Basemap(
	    projection='mill',
	    lat_ts=10,
	    llcrnrlon=lons.min(),
	    urcrnrlon=lons.max(),
	    llcrnrlat=lats.min(),
	    urcrnrlat=lats.max(),
	    resolution='c'
	)
	global_map.drawcoastlines()
	global_map.drawmapboundary()

	global_map.drawparallels(np.arange(-90.,120.,30.), labels=[1,0,0,0], fontsize=8)
	global_map.drawmeridians(np.arange(-180.,180.,60.), labels=[0,0,0,1], fontsize=8)

	x, y = global_map(lons,lats)


def init():
    cs = global_map.pcolormesh(x, y, grb_messages[0].values, shading='flat', cmap=plt.cm.jet)
    plt.colorbar(cs, orientation='vertical', fraction=0.026, pad=0.09)
    txt.set_text(repr(grb_messages[0]))
    return [cs]

def animate(frame):
    cs = global_map.pcolormesh(x, y, grb_messages[frame].values, shading='flat', cmap=plt.cm.jet)
    txt.set_text(repr(grb_messages[frame]))
    return [cs]
    """
	##  07, Use the animation to illustrate the downloaded Hindcast data
	anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(grb_messages), blit=True)
	anim.save('v_wind.mp4')
	"""


# Download, Read, Concert and Save the Hindcast MetOcean data
def ECMWF_retrieve(Hind_start, Hind_end, area_range, area_name):
	"""
	Function:  ECMWF_retrieve(Hind_start, Hind_end, save_grib, save_dat)

		List of Inputs:
			Hind_start [str]: Download the ECMWF data: start date, e.g. '1999-01-01'
			Hind_end   [str]: Download the ECMWF data: end   date, e.g. '1999-01-31'
			area_range [str]: Area of MetOcean to be downloaded, "N/W/S/E, e.g. "90/-80/0/10" means North Atlantic
			area_name  [str]: Give a name for your downloading area

			save_grid  [str]: Folder to save the grid file 
			save_dat   [str]: Folder to save the converted ASCII file


	Here we need to define the following parameters:
		1, The starting day of the Hindcast data downloading
		2, The ending day of the Hindcast data downloading 
		3, The save folder of the Grib2 file from ECMWF
		4, The output folder for ASCII file folder
	All the input will be read from a job file.
	"""
	# Construct the input and output parameters
	dateinterval = Hind_start + '/to/' + Hind_end
	AREA 	     = area_range
	wave_grib_name  = 'wave_' + area_name + Hind_start + '_' + Hind_end + '.grib'
	wind_grib_name  = 'wind_' + area_name + Hind_start + '_' +  Hind_end + '.grib'


	#  Define the sever and download the data
	server = ECMWFDataServer()

	#  Download the wave data, which will include significant wave height Hs, and wave period Tp
	##  01, Define the save folder and file for downloading
	wave_folder = '/Users/wengang/Documents/Python/Spatio_temporal_model/Downloaded_data/GribWaveData/'
	wind_folder = '/Users/wengang/Documents/Python/Spatio_temporal_model/Downloaded_data/GribWindData/'

	##  02, Define the file name and its associated folder
	filename_wave = wave_folder +  wave_grib_name
	filename_wind = wind_folder +  wind_grib_name



	##  03, Download the wave data, i.e. significant wave height, mean wave period, and wave direction
	params =  datasets['significant_wave_height']['param'] + '/' +  datasets['mean_wave_period']['param'] + '/' +  datasets['mean_wave_dir']['param']
	stream = datasets['significant_wave_height']['stream']
	server.retrieve({
	    "class": "ei",
	    "dataset": "interim",
	    "date": dateinterval, #"2014-07-01/to/2014-07-31",
	    "expver": "1",
	    "grid": "0.75/0.75",
	    "repres": "ll",
	    "levtype": "sfc",
	    "param": params,
	    "stream": stream,
	    "time": "00:00:00/06:00:00/12:00:00/18:00:00",
	    "type": "an",
	    "area": AREA,
	    "target": filename_wave,
	})
	print "wind data is downloaded:" + wave_grib_name

	##  04,Download the secondary data, such as the sea temperature, wind_U, wind_V
	params =  datasets['u_wind']['param'] + '/' +  datasets['v_wind']['param']
	#params =  datasets['sea_surface_temp']['param'] + '/' +  datasets['u_wind']['param'] + '/' +  datasets['v_wind']['param']

	stream = datasets['sea_surface_temp']['stream']
	server.retrieve({
	    "class": "ei",
	    "dataset": "interim",
	    "date": dateinterval,  #"2014-07-01/to/2014-07-31",
	    "expver": "1",
	    "grid": "0.75/0.75",
	    "repres": "ll",
	    "levtype": "sfc",
	    "param": params,
	    "stream": stream,
	    "time": "00:00:00/06:00:00/12:00:00/18:00:00",
	    "type": "an",
	    "area": AREA,
	    "target": filename_wind,
	})
	print "wind data is downloaded:" + wind_grib_name



def ECMWF_read_save(Hind_start, Hind_end, area_name, saveif = 0):
	"""
	Function:  ECMWF_read_save(Hind_start, Hind_end, area_name)

		List of Inputs:
			Hind_start [str]: Download the ECMWF data: start date, e.g. '1999-01-01'
			Hind_end   [str]: Download the ECMWF data: end   date, e.g. '1999-01-31'
			area_range [str]: Area of MetOcean to be downloaded, "N/W/S/E, e.g. "90/-80/0/10" means North Atlantic
			area_name  [str]: Give a name for your downloading area

		List of Outputs:


	Here we need to define the following parameters:
		1, The starting day of the Hindcast data downloading
		2, The ending day of the Hindcast data downloading 
		3, The save folder of the Grib2 file from ECMWF
		4, The output folder for ASCII file folder
	All the input will be read from a job file.
	"""


	# Construct the input and output parameters
	dateinterval = Hind_start + '/to/' + Hind_end
	wave_grib_name  = 'wave_' + area_name + Hind_start + '_' + Hind_end + '.grib'
	wind_grib_name  = 'wind_' + area_name + Hind_start + '_' +  Hind_end + '.grib'
	Meto_mat_name = 'Metdata_' + area_name + '_' + Hind_start + '-' +  Hind_end + '.mat'


	#  Read/Load the wave data, which will include significant wave height Hs, and wave period Tp
	##  01, Define the save folder and file for downloading
	wave_folder = '/Users/wengang/Documents/Python/Spatio_temporal_model/Downloaded_data/GribWaveData/'
	wind_folder = '/Users/wengang/Documents/Python/Spatio_temporal_model/Downloaded_data/GribWindData/'
	Meto_folder = '/Users/wengang/Documents/Python/Spatio_temporal_model/Downloaded_data/MatData/'


	##  02, Define the file name and its associated folder
	filename_wave = wave_folder +  wave_grib_name
	filename_wind = wind_folder +  wind_grib_name
	filename_Meto = Meto_folder +  Meto_mat_name

	##  03, Read and extract the wind data 
	### 03.1, To read the wind data	
	pars_wind = pygrib.open(filename_wind)
	wind_U_list = pars_wind.select(name = datasets['u_wind']['name'])
	wind_V_list = pars_wind.select(name = datasets['v_wind']['name'])

	### 03.2, Get the latitute and longtitudes for the hindcast data
	lat, lon    = wind_U_list[0].latlons()

	### 03.3, Pre-define the wind_U and wind_V for the following assignment
	wind_U 		= np.zeros((lat.shape[0], lat.shape[1], len(wind_U_list)), dtype=np.float)
	wind_V 		= np.zeros((lat.shape[0], lat.shape[1], len(wind_U_list)), dtype=np.float)


	### 03.4, Assign the hindcast values of wind_U, wind_V to its values
	################################### NB: you have to check how many parameters saved in the file ######################
	for ind_t in range(len(wind_U_list)):
		wind_U[:,:, ind_t] = wind_U_list[ind_t].values
		wind_V[:,:, ind_t] = wind_V_list[ind_t].values

	##################################################################################################################

	###  04, Read and extract the wave data 
	### 04.1, To read the wind data	
	pars_wave = pygrib.open(filename_wave)
	Hs_list = pars_wave.select(name = datasets['significant_wave_height']['name'])
	Tp_list = pars_wave.select(name = datasets['mean_wave_period']['name'])
	Theta_list = pars_wave.select(name = datasets['mean_wave_dir']['name'])

	### 04.2, Get the latitute and longtitudes for the hindcast data
	lat, lon    = Hs_list[0].latlons()

	### 04.3, Pre-define the Hs, Tp and Theta for the following assignment
	Hs 	= np.zeros((lat.shape[0], lat.shape[1], len(Hs_list)), dtype=np.float)
	Tp 	= np.zeros((lat.shape[0], lat.shape[1], len(Hs_list)), dtype=np.float)
	Theta 	= np.zeros((lat.shape[0], lat.shape[1], len(Hs_list)), dtype=np.float)


	### 04.4, Assign the hindcast values of Hs, Tp and Theta to its values
	################################### NB: you have to check how many parameters saved in the file ######################
	for ind_t in range(len(Hs_list)):
		Hs[:,:, ind_t] 	  =  Hs_list[ind_t].values.data
		Tp[:,:, ind_t] 	  =  Tp_list[ind_t].values.data
		Theta[:,:, ind_t] =  Theta_list[ind_t].values.data

	### 05, Construct the time vector for the Downloaded GRIB data
	delta_T = 0.25  # it is corresponding to 6 hours (temporal interval of hindcast data)
	Data_start  =  datetime2datenum(Hind_start)
	Data_end	=  datetime2datenum(Hind_end + ' 23:00:00')

	Time 		=  np.arange(Data_start, Data_end, delta_T)
	Time 		= Time.reshape(Time.shape[0], 1)

	#### 050, close all files
	pars_wave.close()
	pars_wind.close()

	### 06, Save the files in the system
	#savename = mat_folder + 'Metdata_' + area_name + '_' + str(year) + '.mat'
	if saveif == 1:
		sio.savemat(filename_Meto, dict(Hs=Hs, Tp = Tp, Theta = Theta, wind_U = wind_U, wind_V = wind_V,lon=lon, lat = lat, Time = Time))
		print 'Saving is done!'
	else:
		print 'Loading is done!'

	return Hs, Tp, Theta, wind_U, wind_V, lon, lat, Time



if __name__ == '__main__':
	"""
	In this main function, one should input four "str" elements
	Inputs: 
		Hind_start, i.e. '2010-01-01'
		Hind_end,	i.e. '2010-02-01'
		area_range,  'N/W/S/E', i.e. '65/-90/35/10'
		area_range: a name to related with the above defined region, i.e. 'NorthAtlantic'


	It should be noted that all inputs must be str
	"""

	parser = argparse.ArgumentParser()
	parser.add_argument('Hind_start')
	parser.add_argument('Hind_end')
	parser.add_argument('area_range')
	parser.add_argument('area_name')
	#parser.add_argument('save_dat')
	args = parser.parse_args()
	ECMWF_retrieve(args.Hind_start, args.Hind_end, args.area_rang, args.area_name)  #, args.save_dat)
	ECMWF_read_save(args.Hind_start, args.Hind_end, args.area_name)

