# standard libs
import os
import re
import sys
from functools import reduce

# third party libs
import numpy as np
import pandas as pd
import lmfit as lf
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import plotly.express as px
#from matplotlib.figure import Figure

# custom modules
from grblc.util import get_dir
from . import io


class Lightcurve: # define the object Lightcurve
    _name_placeholder = "unknown grb" # assign the name for GRB if not provided
    _flux_fixed_inplace = False #


    def __init__(
        self,
        path: str = None,
        appx_bands: str = True, # if it is True it enables the approximation of bands, e.g. u' approximated to u,.....
        name: str = None,
    ):
        """The main module for fitting lightcurves.

        Parameters
        ----------
        path : str, optional
            Name of file containing light curve data, by default None
        xdata : array_like, optional
            X values, length (n,), by default None
        ydata : array_like, optional
            Y values, by default None
        xerr : array_like, optional
            X error, by default None
        yerr : array_like, optional
            Y error, by default None
        data_space : str, {log, lin}, optional
            Whether the data inputted is in log or linear space, by default 'log'
        name : str, optional
            Name of the GRB, by default :py:class:`Model` name, or ``unknown grb`` if not
            provided.
        """
        #assert bool(path) ^ (
        #    xdata is not None and ydata is not None
        #), "Either provide a path or xdata, ydata."


        # some default conditions for the name of GRBs and the path of the data file
        if name:
            self.name = name  # asserting the name of the GRB
        else:
            self.name = self._name_placeholder  # asserting the name of the GRB as 'Unknown GRB' if the name is not provided

        if isinstance(path, str):
            self.path = path  # asserting the path of the data file
            self.set_data(path, appx_bands=appx_bands, data_space='lin') # reading the data from a file


    def set_data(self, path: str, appx_bands=True, data_space='lin'):
        """
            Reads in data from a file. The data must be in the correct format.
            See the :py:meth:`io.read_data` for more information.

            Set the `xdata` and `ydata`, and optionally `xerr` and `yerr` of the lightcurve.

        .. warning::
            Data stored in :py:class:`lightcurve` objects are always in logarithmic
            space; the parameter ``data_space`` is only used to convert data to log space
            if it is not already in such. If your data is in linear space [i.e., your
            time data is sec, and not log(sec)], then you should set ``data_space``
            to ``lin``.

        Parameters
        ----------
        xdata : array_like
            X data
        ydata : array_like
            Y data
        xerr : array_like, optional
            X error, by default None
        yerr : array_like, optional
            Y error, by default None
        data_space : str, {log, lin}, optional
            Whether the data inputted is in logarithmic or linear space, by default 'log'.
        """

        df = io.read_data(path) # reads the data, sorts by time, excludes negative time

        df = df[df['mag_err'] != 0] # asserting those data points only which does not have limiting nagnitude
        assert len(df)!=0, "Only limiting magnitudes present."

        # converting the data here in the required format for color evolution analysis
        def convert_data(data):

            data = list(data) # reading the data as a list

            for i, band in enumerate(data):
                if band.lower() in ['clear', 'unfiltered', 'lum']:  # here it is checking for existence of the bands in lower case for three filters 'clear', 'unfiltered', 'lum'
                    band == band.lower()  # here it passes the lower case bands

            if appx_bands:  # here we reassigns the bands (reapproximation of the bands), e.g. u' reaasigned to u,.....
                for i, band in enumerate(data):
                    if band=="u'":
                        data[i]="u"
                    if band=="g'":
                        data[i]="g"
                    if band=="r'":
                        data[i]="r"
                    if band=="i'":
                        data[i]="i"
                    if band=="z'":
                        data[i]="z"
                    if band.upper()=="BJ":
                        data[i]="B"
                    if band.upper()=="VJ":
                        data[i]="V"
                    if band.upper()=="UJ":
                        data[i]="U"
                    if band.upper()=="RM":
                        data[i]="R"
                    if band.upper()=="BM":
                        data[i]="B"
                    if band.upper()=="UM":
                        data[i]="U"
                    if band.upper()=="JS":
                        data[i]="J"
                    if band.upper()=="KS":
                        data[i]="K"
                    if band.upper()=="K'":
                        data[i]="K"
                    if band.upper()=="KP":
                        data[i]="K"

                bands = data
            else:
                bands = data

            return bands


        self.xdata = df["time_sec"].to_numpy()  # passing the time in sec as a numpy array in the x column of the data
        self.ydata = df["mag"].to_numpy() # passing the magnitude as a numpy array in the y column of the data
        self.yerr = df["mag_err"].to_numpy()  # passing the magnitude error as an numpy array y error column of the data
        self.band_og = df["band"].to_list() # passing the original bands (befotre approximation of the bands) as a list
        self.band = df["band"] = convert_data(df["band"]) # passing the reassigned bands (after the reapproximation of the bands) as a list
        self.system = df["system"].to_list()  # passing the filter system as a list
        self.telescope = df["telescope"].to_list()  # passing the telescope name as a list
        self.extcorr = df["extcorr"].to_list()  # passing the galactic extinction correction detail (if it is corrected or not) as a list
        self.source = df["source"].to_list()  # passing the source from where the particular data point has been gathered as a list
        self.df = df  # passing the whole data as a data frame
      
    def displayGRB(self, save_static=False, save_static_type='.png', save_interactive=False, save_in_folder='plots/'):
        # This function plots the magnitudes, excluding the limiting magnitudes

        '''
        For an interactive plot
        '''

        fig = px.scatter(
                    x=self.xdata,
                    y=self.ydata,
                    error_y=self.yerr,
                    color=self.band,
                    hover_data=self.telescope,
                )

        font_dict=dict(family='arial',
                    size=18,
                    color='black'
                    )

        fig['layout']['yaxis']['autorange'] = 'reversed'
        fig.update_yaxes(title_text="<b>Magnitude<b>",
                        title_font_color='black',
                        title_font_size=18,
                        showline=True,
                        showticklabels=True,
                        showgrid=False,
                        linecolor='black', 
                        linewidth=2.4, 
                        ticks='outside', 
                        tickfont=font_dict,
                        mirror='allticks', 
                        tickwidth=2.4, 
                        tickcolor='black',  
                        )

        fig.update_xaxes(title_text="<b>log10 Time (s)<b>",
                        title_font_color='black',
                        title_font_size=18,
                        showline=True,
                        showticklabels=True,
                        showgrid=False,
                        linecolor='black',
                        linewidth=2.4,
                        ticks='outside',
                        tickfont=font_dict,
                        mirror='allticks',
                        tickwidth=2.4,
                        tickcolor='black',
                        )

        fig.update_layout(title="GRB " + self.name,
                        title_font_size=25,
                        font=font_dict,
                        plot_bgcolor='white',  
                        width=960,
                        height=540,
                        margin=dict(l=40,r=40,t=50,b=40)
                        )

        if save_static:
            fig.write_image(save_in_folder+self.name+save_static_type)

        if save_interactive:
            fig.write_html(save_in_folder+self.name+'.html')

        return fig

    def colorevolGRB(self, print_status=True, return_rescaledf=False, save_plot=False, chosenfilter='mostnumerous', save_in_folder=''): #, rescaled_dfsave=False):

        global compzerolist, nocompzerolist, overlap

        def overlap(mag1lower,mag1upper,mag2lower,mag2upper):
            if mag1upper <mag2lower or mag1lower > mag2upper:
                return 0 # no overlap
            else:
                return max(mag1upper, mag2upper) # 

        light = pd.DataFrame()
        light['time_sec'] = self.xdata # time is linear
        light['mag'] = self.ydata
        light['mag_err'] = self.yerr
        light['band'] = self.band

        #print(light)

        light = light[(light['mag_err']!=0) & (light['time_sec']>0)] # here I require only magnitudes and not limiting magnitudes,
                                                                     # there are some transients observed in the optical before the 
                                                                     # satellite trigger, thus they have negative times since in our
                                                                     # analysis, we consider the trigger time as start time of the LC

        assert len(light)!=0, "Has only limiting magnitudes." # assert is a command that verifies the condition written, if the condition
                                                              # doesn't hold then the error on the right is printed
        assert len(light)>1, "Has only one data point."       # here we highlight if the dataframe has only limiting magnitudes
                                                              # or if it has only one data point

        occur = light['band'].value_counts()    # this command returns a dataframe that contains in one column the
                                                # label of the filter and in another column the occurrences
                                                # Example: filter occurrences
                                                # R 18
                                                # I 6
                                                # B 5
                                                # F606W 4

        # Identifying the most numerous filter in the GRB 
        
        assert chosenfilter == 'mostnumerous' or filterforrescaling in self.band, "Rescaling band provided is not present in data!"
        
        # chosenfilter is an input of the function colorevolGRB(...)
        # This assert condition is needed to verify that the filter is present in the LC; for example, if "g" is selected but it's 
        # not present then the string "Rescaling..." is printed

        if chosenfilter == 'mostnumerous':          # here I select by default the filterforrescaling as the most numerous inside the LC 
                                                    # if chosenfilter input is 'mostnumerous', then I automatically take the most numerous
            filterforrescaling = occur.index[0]     # namely, the first element in the occur frame (with index 0, since in Python the counting starts from zero)
            filterforrescaling_occur = occur[0]     # this is the number of occurrences of the filterforrescaling
        else:
            for ii in occur.index:                  # if the input chosenfilter of the function is a filter label different from 'mostnumerous'
                if ii==chosenfilter:                # with this loop, the input variable chosenfilter is matched with the list of filters called "occur"
                    filterforrescaling = ii
                    filterforrescaling_occur = occur[ii]

        if print_status:                            # the print_status option is set to true by default, and it prints
            print(self.name)                        # the GRB name
            print('\n')
            print('-------')                        # and the details of the filter chosen for rescaling, name + occurrences
            print(occur, 'The filter chosen in this GRB: ',filterforrescaling,', with', filterforrescaling_occur, 'occurrences.\n'+
                'This filter will be considered for rescaling')
        
        
        # The scalingfactorslist is crucial: for every filter of the GRB, it contains the elements in the following form
        # [filter,occurrences,[[linear_time_filter1,rescalingfactor_to_filterforrescaling1,linear_timedifference_betweenfilter_and_filterforrescaling1,rescalingfactorerr1],
        # [linear_time_filter2,(...)],(...)]] -> given a filter, there may be more rescaling factors to the filterforrescaling
        
        scalingfactorslist = [] 
        
        mostcommonlight=light.loc[(light['band'] == filterforrescaling)] # mostcommonlight dataframe is the one constituted of the chosen filter for rescaling, 
        mostcommonx=mostcommonlight['time_sec'].values                   # either the most numerous by default or the filterforrescaling put in the function
        mostcommony=mostcommonlight['mag'].values                        # time_sec is linear  
        mostcommonyerr=mostcommonlight['mag_err'].values  
        
        for j in range(len(occur)):
            if occur.index[j]!=filterforrescaling:                          # here we prepare the other entries for the scalingfactorslist
                scalingfactorslist.append([occur.index[j],occur[j],[]])     # we specify the filter name and the occurrences, and the empty list (the third element inside)
                                                                        # will be filled with linear_time_filter,rescalingfactor_to_filterforrescaling,linear_timedifference_betweenfilter_and_filterforrescaling,rescalingfactorerr

        for j in range(len(scalingfactorslist)): # this loop skips the first element, namely the filter chosen for rescaling; it loops on all the other filters of the GRB
            
            sublight=light.loc[(light['band'] == scalingfactorslist[j][0])] # the sublight is the dataframe where only the magnitudes of one filter are considered
            subx=sublight['time_sec'].values    # time_sec is linear
            suby=sublight['mag'].values
            subyerr=sublight['mag_err'].values
            #subcolor=sublight['band'].values

            # the timediffcoinc is a list of indices [p1,p2]: p1 runs on the dataframe of most numerous filter,
            # while p2 runs on the sublight frame of the filter on which the loop iterates
            # The condition here requires to select the points [p1(filterforrescaling),p2(anotherfilter)] such that
            # the linear time of p1 and the linear time of p2 are equal

            timediffcoinc = [[p1,p2] for p1 in range(len(mostcommonx)) for p2 in range(len(subx))
                        if mostcommonx[p1]==subx[p2]]

            # The timediff list is the same as above, the only difference is that here p1 and p2 are selected if
            # | linear_time(p1) - linear_time(p2) | / linear_time(p1) <= 2.5 %

            timediff = [[p1,p2] for p1 in range(len(mostcommonx)) for p2 in range(len(subx))
                        if np.abs(mostcommonx[p1]-subx[p2])<=((mostcommonx[p1])*0.025)]
            
            # Why creating the lists timediffcoinc and timediff? Because, for the rescaling factors estimation, the priority
            # is given to the filters at the exact same time; if these are not present, then we select the condition for which
            # the time difference is <= 2.5% of the time of the filter chosen for rescaling. This is expressed in the following
            # if condition:
            
            if len(timediffcoinc)!=0:       # when we have the rescaling factors from the same exact times
                for ll in timediffcoinc:    # here we create the elements to be appended in the scalingfactorslist empty third elements (see line 441)
                    sf2=[subx[ll[1]],       # linear_time_filter          
                        mostcommony[ll[0]]-suby[ll[1]], # rescalingfactor_to_filterforrescaling
                        np.abs(mostcommonx[ll[0]]-subx[ll[1]]), # linear_timedifference_betweenfilter_and_filterforrescaling 
                        np.sqrt(mostcommonyerr[ll[0]]**2+subyerr[ll[1]]**2)] # rescalingfactorerr
                    scalingfactorslist[j][2].append(sf2)  

            # Here the loop is the same of above, it is called only if the coicident times between the filter for rescaling and other
            # filters are not present, but the times in the 2.5% condition are present instead

            elif len(timediffcoinc)==0 and len(timediff)!=0:
                for ll in timediff:
                    sf2=[subx[ll[1]],
                        mostcommony[ll[0]]-suby[ll[1]],
                        np.abs(mostcommonx[ll[0]]-subx[ll[1]]),
                        np.sqrt(mostcommonyerr[ll[0]]**2+subyerr[ll[1]]**2)]
                    scalingfactorslist[j][2].append(sf2)  

            else:           # in the remaining cases, included by the else condition, for the filter in the loop there is no rescaling factor to the filter for rescaling
                continue

        # The loop that follows is crucial for the color evolution analysis. The color evolution will require a plot of
        # rescaling factors vs log10(time_linear). It may happen that, given a filter at a time - say - t1, 
        # there may be different filters chosen for rescaling that are in the 2.5% condition with the filter at t1. 
        # So, how to avoid to have different rescaling factors for a filter at the same time (and, thus, multiple y values for the same x in the plot)?
        # Among the points of a filter chosen for rescaling that are in the 2.5% condition with the filter at t1, 
        # we select only the one that has the minimum time distance with the t1.

        evolutionrescalingfactor=[] # here the evolutionrescalingfactor list is initialized; it will include the elements useful for the color evolution fitting 

        for fl in scalingfactorslist: # the loop is on the list of rescaling factors

            times=set(el[0] for el in fl[2]) # here the code takes only the linear_time_filter for each of the rescaling factors inside the scalingfactorslist
            
            for tt in times:    # the loop goes on each of the linear_time_filter, say t1
                suppllist=[fl[2][x] for x in range(len(fl[2])) if fl[2][x][0]==tt] # this supplementary list contains all the rescaling factors that are estimated at the t1
                suppllistdist=[fl[2][x][2] for x in range(len(fl[2])) if fl[2][x][0]==tt]   # this supplementary list contains all the time differences between the t1 and the times of the most numerous filters 
                                                                                            # for the rescaling factors estimated at t1     
                
                mindistpos=suppllistdist.index(min(suppllistdist)) # this command finds the position of the minimum of the suppllistdist, namely, 
                                                                   # points out which rescaling factor compatible with the filter at time t1 has the minimum time distance from it 
                
                evolutionrescalingfactor.append([fl[0],fl[1],suppllist[mindistpos]]) # in the evolutionrescalingfactor list we append only one rescaling factor for each t1, 
                                                                                     # and this is the one with the smallest time difference between t1 and the time of the filter chosen for rescaling   
                
        # The finalevolutionlist is the list evolutionrescalingfactor sorted for the time (t1) and will be used for the
        # resc fact vs log10(time) linear fitting and plotting
        finalevolutionlist=sorted(evolutionrescalingfactor, key=lambda finalevolutionlist: finalevolutionlist[2][0])

        filt=[jj[0] for jj in finalevolutionlist if jj[0]!=filterforrescaling] # here we have the filters that are rescaled to the selected filter for rescaling
        filtoccur=[jj[1] for jj in finalevolutionlist if jj[0]!=filterforrescaling] # here we have the occurrences of the filters
        resclogtime=[np.log10(jj[2][0]) for jj in finalevolutionlist if jj[0]!=filterforrescaling] # WATCH OUT! For the plot and fitting, we take the log10(time) of rescaling factor
        rescfact=[jj[2][1] for jj in finalevolutionlist if jj[0]!=filterforrescaling] # The rescaling factor value
        rescfacterr=[jj[2][3] for jj in finalevolutionlist if jj[0]!=filterforrescaling] # The rescaling factor error
        rescfactweights=[(1/jj[2][3]) for jj in finalevolutionlist if jj[0]!=filterforrescaling] # The weights on the rescaling factor
        
        # The following command defines the dataframe of rescaling factors

        rescale_df=pd.DataFrame(list(zip(filt,filtoccur,resclogtime,rescfact,
                                                    rescfacterr,rescfactweights)),columns=['band','Occur_band','Log10(t)','Resc_fact','Resc_fact_err','Resc_fact_weights'])

        x_all = rescale_df['Log10(t)']  # list of log10 times for the rescaling factors
        y_all = rescale_df['Resc_fact'] # list of the rescaling factors
        yerr_all = rescale_df['Resc_fact_err'] # list of the rescaling factors errors
        filters = [*set(rescale_df['band'].values)] # list of filters in the rescaling factors sample
        rescale_df['plot_color'] = "" # empty list that will filled with the color map condition

        # Set the color map to match the number of filter
        cmap = plt.get_cmap('gist_ncar') # import the color map
        cNorm  = colors.Normalize(vmin=0, vmax=len(filters)) # linear map of the colors in the colormap from data values vmin to vmax
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap) # The ScalarMappable applies data normalization before returning RGBA colors from the given colormap

        # Plot each filter
        fig = plt.figure()

        for i, band in enumerate(filters): # loop on the given filter
            colour = scalarMap.to_rgba(i) # mapping the colour into the RGBA
            index = rescale_df['band'] == band # selects the magnitudes that have the filter equal to the band on which the loop iterates
            plt.scatter(x_all[index], y_all[index], # options for the plot of the central values
                        s=15, 
                        color=colour) # color-coding of the plot
            plt.errorbar(x_all[index], y_all[index], yerr_all[index], #options for the plot of the error bars, these must be added in this command
                        fmt='o', # this is the data marker, a circle
                        barsabove=True, # bars plotted above the data marker
                        ls='', # line style = None, so no lines are drawn between the points (later the fit will be done and plotted)
                        color=colour # color-coding
                        )
            for j in rescale_df[index].index:
                rescale_df.at[j,"plot_color"] = colour # this loop assigns each filter to a color in the plot

        resc_slopes_df = pd.DataFrame() # initialize of the rescaling factors fitting dataframe
        resc_slopes_df.index = filters # the filters are taken as index
        resc_slopes_df['slope'] = "" # placeholder, default set to empty, then it will change - slope of the linear fit
        resc_slopes_df['slope_err'] = "" # placeholder, default set to empty, then it will change - error on slope
        resc_slopes_df['intercept'] = "" # placeholder, default set to empty, then it will change - intercept of linear fit
        resc_slopes_df['inter_err'] = "" # placeholder, default set to empty, then it will change - error on intercept
        resc_slopes_df['acceptance'] = "" # placeholder, default set to empty, then it will change - acceptance = |slope_err|/|slope|
        resc_slopes_df['red_chi2'] = "" # placeholder, default set to empty, then it will change - reduced chi^2
        resc_slopes_df['comment'] = "" # placeholder, default set to empty, then it will change - the comment that will say "no color evolution","color evolution"
        resc_slopes_df['plot_color'] = "" # placeholder, default set to empty, then it will change - color-coding for the fitting lines

        for band in resc_slopes_df.index: # in this loop, we assign the bands in the dataframe defined in line 580
            ind = rescale_df.index[rescale_df['band'] == band][0] 
            resc_slopes_df.loc[band]['plot_color'] = rescale_df.loc[ind]["plot_color"]
            resc_band_df = rescale_df[rescale_df['band'] == band]

            x = resc_band_df['Log10(t)'] # we here define the dataframe to fit, log10(time)
            y = resc_band_df['Resc_fact'] # the rescaling factors
            weights = resc_band_df['Resc_fact_weights'] # the rescaling factors weights
            
            ## lmfit linear - lmfit is imported as "lf" -> the lmfit uses the Levenberg-Marquardt method 
            # https://lmfit.github.io/lmfit-py/

            if len(x) >= 3: # the fitting will be performed if and only if, for the given filter, at least 3 rescaling factors are available
                linear_model = lf.models.LinearModel(prefix='line_') # importing linear model from lmfit
                linear_params = linear_model.make_params() # we here initialize the fitting parameters, then these will be changed
                
                linear_params['line_slope'].set(value=-1.0) # initializing the fitting slope
                linear_params['line_intercept'].set(value=np.max(y)) # initializing the fitting intercept

                linear_fit = linear_model.fit(y, params=linear_params, x=x, weights=weights) # the command for weighted lmfit
                
                resc_slopes_df.loc[band]['slope'] = np.around(linear_fit.params['line_slope'].value, decimals=4) # slope of the fitting
                resc_slopes_df.loc[band]['slope_err'] = np.around(linear_fit.params['line_slope'].stderr, decimals=4) # slope error
                resc_slopes_df.loc[band]['intercept'] = np.around(linear_fit.params['line_intercept'].value, decimals=4) # intercept
                resc_slopes_df.loc[band]['inter_err'] = np.around(linear_fit.params['line_intercept'].stderr, decimals=4) # intercept error
                resc_slopes_df.loc[band]['acceptance'] = np.around(np.abs(resc_slopes_df.loc[band]['slope_err']/resc_slopes_df.loc[band]['slope']), decimals=4) # acceptance = |slope_err|/|slope|
                resc_slopes_df.loc[band]['red_chi2'] = np.around(linear_fit.redchi, decimals=4) # reduced chi^2
                
            else: # not enough data points, less than 3 rescaling factors for the filter
                resc_slopes_df.loc[band]['slope'] = np.nan 
                resc_slopes_df.loc[band]['slope_err'] = np.nan
                resc_slopes_df.loc[band]['intercept'] = np.nan
                resc_slopes_df.loc[band]['inter_err'] = np.nan
                resc_slopes_df.loc[band]['acceptance'] = np.nan
                resc_slopes_df.loc[band]['comment'] = "insufficient data"
                resc_slopes_df.loc[band]['red_chi2'] = 'insufficient data'
                
            if resc_slopes_df.loc[band]['slope'] != 0: # in the case of non-zero slope
                if resc_slopes_df.loc[band]['acceptance'] < 10000: # this boundary of acceptance is put ad-hoc to show all the plots

                    y_fit = resc_slopes_df.loc[band]['slope'] * x + resc_slopes_df.loc[band]['intercept'] # fitted y-value according to linear model

                    plt.plot(x, y_fit, 
                            color=resc_slopes_df.loc[band]["plot_color"]) # plot of the fitting line between log10(t) and resc_fact

                    if np.abs(resc_slopes_df.loc[band]['slope']) < 0.1: # in case the |slope| is smaller than 0.1 then there is no color evolution
                        resc_slopes_df.loc[band]['comment'] = "no color evolution"
                        # the following condition checks if the slope is compatible with zero in 3 sigma
                    elif resc_slopes_df.loc[band]['slope']-(3*resc_slopes_df.loc[band]['slope_err'])<=0<=resc_slopes_df.loc[band]['slope']+(3*resc_slopes_df.loc[band]['slope_err']):
                        resc_slopes_df.loc[band]['comment'] = "no color evolution" # in case it is comp. with zero in 3 sigma, there is no color evolution
                    else:    
                        resc_slopes_df.loc[band]['comment'] = "slope >= 0.1" # in case the slope is not compatible with zero in 3 sigma

                else:
                    resc_slopes_df.loc[band]['comment'] = "slope=nan"  # when the slope is not estimated in the fitting (no compatibility with zero in 3sigma)

        for band in resc_slopes_df.index: # this loop defines the labels to be put in the rescaling factor plot legend
            
            if np.isnan(resc_slopes_df.loc[band]["slope"])==True: # in case the fitting is not done, the label will be "filter: no fitting"
                label=band+": no fitting"
            else:
                label=band+": "+ str(resc_slopes_df.loc[band]["slope"]) + r'$\pm$' + str(resc_slopes_df.loc[band]["slope_err"])
                # when the slopes are estimated, the label is "filter: slope +/- slope_err" 

            ind = rescale_df.index[rescale_df['band'] == band][0] # initializing the variables to be plotted for each filter
            color = rescale_df.loc[ind]["plot_color"]
            plt.scatter(x=[], y=[], 
                        color=color, 
                        label=label # here the labels for each filter are inserted
                        )
    
        plt.rcParams['legend.title_fontsize'] = 'xx-large' #options for the plot of rescaling factors
        plt.xlabel('Log time (s)',fontsize=22)
        plt.ylabel('Rescaling factor with respect to '+filterforrescaling+' (mag)',fontsize=22)
        plt.rcParams['figure.figsize'] = [15, 10]
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.title("GRB "+self.name, fontsize=22)
        plt.legend(title='Band & slope', bbox_to_anchor=(1.015, 1.015), loc='upper left', fontsize='xx-large') # legend, it uses the colors and labels  

        if save_plot:
            plt.savefig(os.path.join(save_in_folder+'/'+str(self.name)+'_colorevol.pdf'), dpi=300) # option to export the pdf plot of rescaling factors

        plt.show()

        # Here the code prints the dataframe of rescaling factors, that contains log10(time), slope, slope_err...
        rescale_df.drop(labels='plot_color', axis=1, inplace=True)     # before printing that dataframe, the code removes the columns of plot_color
        resc_slopes_df.drop(labels='plot_color', axis=1, inplace=True) # since this column was needed only for assigning the plot colors
                                                                       # these columns have no scientific meaning
        
        if print_status: # when this option is selected in the function it prints the following

            print("Individual point rescaling:")
            print(rescale_df) # the dataframe of rescaling factors

            print("\nSlopes of rescale factors for each filter:")
            print(resc_slopes_df) # the dataframe that contains the fitting parameters of rescaling factors
             
            compatibilitylist=[] # here we initialize the list that contains the ranges of (slope-3sigma,slope+3sigma) for each filter
    
            for band in resc_slopes_df.index: # this code appends the values of (slope-3sigma,slope+3sigma) in case the slope is not a "nan"
                if resc_slopes_df.loc[band]['slope']!=0 and resc_slopes_df.loc[band]['slope_err']!=0 and np.isnan(resc_slopes_df.loc[band]['slope'])==False and np.isnan(resc_slopes_df.loc[band]['slope_err'])==False:
                    compatibilitylist.append([band,[resc_slopes_df.loc[band]['slope']-(3*resc_slopes_df.loc[band]['slope_err']),
                                            resc_slopes_df.loc[band]['slope']+(3*resc_slopes_df.loc[band]['slope_err'])]])

            compzerolist=[] # this is the list of filters with slopes that are compatible with zero in 3 sigma
            nocompzerolist=[] # this is the list of filters with slopes that ARE NOT compatible with zero in 3 sigma
            for l in compatibilitylist:
                if l[1][0]<=0<=l[1][1] or np.abs((l[1][0]+l[1][1])/2)<0.10: # if slope-3sigma<=0<=slope+3sigma (namely, compat. with zero in 3sigma)
                        compzerolist.append(l[0])                           # or if the slope in absolute value is smaller than 0.10
                else:                                                       # then for the given filter the slope is compatible with zero in 3sigma, NO COLOR EVOLUTION
                    nocompzerolist.append(l[0]) # in the other cases, slopes are not compatible with zero in 3 sigma, COLOR EVOLUTION

            if len(compzerolist)==0: # if there are no filters compatible with zero in 3 sigma
                print('No filters compatible with zero in 3sigma or with |slope|<0.1')
                
            else:
                print('Filters compatible with zero in 3sigma: ',*compzerolist) # if there are filters without color evolution, namely, compatible with zero in 3 sigma
            
            if len(nocompzerolist)==0: # if there are not filters with color evolution, namely, that are not compatible with zero in 3 sigma
                print('No filters with |slope|>0.1 or compatible with zero only in >3sigma')
                
            else: # otherwise
                print('Filters not compatible with zero in 3sigma or with |slope|>0.1: ',*nocompzerolist)    

            print('\n')
            print('No color evolution: ',*compzerolist,' ; Color evolution: ',*nocompzerolist) # print of the two lists       

            
            string="" # this is the general printing of all the slopes
            for band in resc_slopes_df.index:
                string=string+band+":"+str(round(resc_slopes_df.loc[band]['slope'],3))+"+/-"+str(round(resc_slopes_df.loc[band]['slope_err'],3))+"; "
                
            print(string)

        if return_rescaledf: # variables returned in case the option return_rescaledf is enabled
            return fig, rescale_df, resc_slopes_df, scalingfactorslist, compzerolist, nocompzerolist, filterforrescaling

        return fig, resc_slopes_df, scalingfactorslist, compzerolist, nocompzerolist, filterforrescaling # the variables in the other case


    def rescaleGRB(self, chosenfilter='mostnumerous', save_rescaled_in=''): # this function makes the rescaling of the GRB

        # the global option is needed when these variables inputed in the current function are output of another function recalled, namely, colorevolGRB
        global compzerolist, scalingfactorslist, filterforrescaling 

        # here the code uses the colorevolGRB function defined above
        input = self.colorevolGRB(print_status=False, return_rescaledf=False, save_plot=False, chosenfilter=chosenfilter, save_in_folder=save_rescaled_in)

        scalingfactorslist = input[2] # this is the scaling factors list, needed for the rescaling        
        compzerolist = input[3] # this is the list of filters whose resc.fact. slopes are compatible with zero in 3sigma or are < 0.10 
        filterforrescaling = input[5] # this is the filter chosen for rescaling

        lightresc = pd.DataFrame() # here the code imports the original dataframe, that is needed for the final rescaled dataframe
        lightresc['time_sec'] = self.xdata # time is linear
        lightresc['mag'] = self.ydata
        lightresc['mag_err'] = self.yerr
        lightresc['band'] = self.band
        lightresc['band_og'] = self.band_og
        lightresc['system'] = self.system
        lightresc['telescope'] = self.telescope
        lightresc['extcorr'] = self.extcorr
        lightresc['source'] = self.source

        # Before rescaling the magnitudes, the following instructions plot the magnitudes in the unrescaled case
        figunresc = px.scatter(
                x=np.log10(lightresc['time_sec'].values), # the time is set to log10(time) only in the plot frame
                y=lightresc['mag'].values,
                error_y=lightresc['mag_err'].values,
                color=lightresc['band'].values,
                )

        font_dict=dict(family='arial',
                    size=18,
                    color='black'
                    )

        figunresc['layout']['yaxis']['autorange'] = 'reversed'
        figunresc.update_yaxes(title_text="<b>Magnitude<b>",
                        title_font_color='black',
                        title_font_size=18,
                        showline=True,
                        showticklabels=True,
                        showgrid=False,
                        linecolor='black', 
                        linewidth=2.4, 
                        ticks='outside', 
                        tickfont=font_dict,
                        mirror='allticks', 
                        tickwidth=2.4, 
                        tickcolor='black',  
                        )

        figunresc.update_xaxes(title_text="<b>log10 Time (s)<b>",
                        title_font_color='black',
                        title_font_size=18,
                        showline=True,
                        showticklabels=True,
                        showgrid=False,
                        linecolor='black',
                        linewidth=2.4,
                        ticks='outside',
                        tickfont=font_dict,
                        mirror='allticks',
                        tickwidth=2.4,
                        tickcolor='black',
                        )

        figunresc.update_layout(title="GRB " + self.name,
                        title_font_size=25,
                        font=font_dict,
                        plot_bgcolor='white',  
                        width=960,
                        height=540,
                        margin=dict(l=40,r=40,t=50,b=40)
                        )

        figunresc.show()

        averagedrescalingfactor = [] # this is the list that will contain the averaged rescaling factors for each filter

        for c in scalingfactorslist: # this starts from 1 since it excludes the filter used for rescaling (that is the first element of the list)
                
            # To make the weighted average, we used the instructions in this page (extracted from Taylor textbook for errors analysis)
            # https://www.amherst.edu/system/files/media/1871/weighted%2520average.pdf
                
            if len(c[2])!=0: # for each filter, if the list of rescaling factors is not empty
                rescfactfilter=[c[2][x][1] for x in range(len(c[2]))] # rescaling factor
                rescfacterrfilter=[c[2][x][3] for x in range(len(c[2]))] # error on rescaling factor
                rescfactweightfilter=[1/(rr**2) for rr in rescfacterrfilter] # weight on rescaling factor
                # weighted mean of rescaling factor
                meanrescfact=sum([rescfactweightfilter[k]*rescfactfilter[k] for k in range(len(rescfactfilter))])/sum(rescfactweightfilter)
                # error on weighted mean of rescaling factor
                meanrescfacterr=1/np.sqrt(sum(rescfactweightfilter))
                
            else: # if the list is empty, then the rescaling factor variables are set all to zero
                rescfactfilter=[0]
                rescfacterrfilter=[0]
                rescfactweightfilter=[0]
                meanrescfact=0
                meanrescfacterr=0

            # this is a safety option, if there are some divergences in the quantities in the rescaling factors weighted mean
            # namely, if the weighted mean rescaling factor and its error are NaN
            # then the weighted mean rescaling factor and its error are set to zero
            if np.isnan(np.mean(rescfactfilter))==False: 
                averagedrescalingfactor.append([c[0],c[1],meanrescfact,meanrescfacterr])
            if np.isnan(np.mean(rescfactfilter))==True:
                averagedrescalingfactor.append([c[0],c[1],0,0]) 

        # The lightresc frame is the one that will be rescaled

        # initializing the lightresc frame, exporting the list of values from the lightresc dataframe
        lightrescx = lightresc['time_sec'].values # here the time is linear
        lightrescy = lightresc['mag'].values # this will be rescaled according to the rescaling factor, 2.5% of time and magnitudes overlap condition
        lightrescyerr = lightresc['mag_err'].values 
        lightresccolor = lightresc['band'].values
        lightrescsystem = lightresc['system'].values
        lightresctelescope = lightresc['telescope'].values
        lightrescextcorr = lightresc['extcorr'].values
        lightrescsource = lightresc['source'].values
            
        for pp in range(len(lightresc)): # this loop runs on every point of the lightcurve to be rescaled
            mag2lower=lightrescy[pp]-lightrescyerr[pp] # for the point, this is magnitude-1sigma
            mag2upper=lightrescy[pp]+lightrescyerr[pp] # for the point, this is magnitude+1sigma
            supportlistov=[] # this list will contain the overlap of the magnitudes values, 0 if no overlap, nonzero otherwise
            supportlistdist=[] # this list will contain the time difference between the filter chosen for rescaling and the filter from the loop that must be rescaled
            
            
            for ff in averagedrescalingfactor: # this loop runs on the mean rescaling factors list for each filter
                if lightresccolor[pp] == ff[0]: # matching the filter in the averagedrescalingfactor with the one to be rescaled
                    # the following loop runs on all the points that have the filter used for rescaling (usually, the most numerous)
                    # x != pp is for safety, we avoid using the same point to be compared with itself
                    for nn in [x for x in range(len(lightresc)) if (x != pp) & (lightresccolor[x]==filterforrescaling)]: 
                        mag1lower=lightrescy[nn]-lightrescyerr[nn] # for the point of the filter used for rescaling, this is magnitude-1sigma
                        mag1upper=lightrescy[nn]+lightrescyerr[nn] # for the point of the filter used for rescaling, this is magnitude+1sigma
                        # this estimated the time difference: |time(rescaling_reference_filter) - time(filter_to_be_rescaled)|/time(rescaling_reference_filter)
                        dist=np.abs((lightrescx[nn])-(lightrescx[pp]))/(lightrescx[nn]) 
                        # this command uses the overlap function defined above to estimate the overlap between the point to be rescaled
                        # and the filter used for rescaling; then, it appends to supportlistov the overlap (0 if no overlap, non-zero if overlap)
                        supportlistov.append(overlap(mag1lower, mag1upper, mag2lower, mag2upper))
                        # this last command in the loop appends to the supportlist the time difference "dist" defined in line 944
                        supportlistdist.append(dist)
                        
                    # The condition for rescaling applies
                    # The check is done among all the possible filters chosen for rescaling and the point that must be rescaled
                    # If, for any of these filters chosen for rescaling, the following conditions hold
                    # 1) The overlap among magnitudes of the filter chosen for rescaling and the magnitudes of the point to be rescaled is ZERO (no overlap)
                    # 2) The time distance between the linear times of the filter chosen for rescaling and the point to be rescaled is smaller than the 2.5% of time
                    # of the filter chosen for rescaling
                    # 3) If the filter has no color evolution (i.e. it belongs to the compzerolist)
                    # IF ALL OF THESE 3 HOLD, then the rescaling if performed
                    if any((supportlistov[i] == 0) & (supportlistdist[i] <= 0.025) & (lightresccolor[pp] in compzerolist) for i in range(len(supportlistov))):
                        # the magnitude of the filter to be rescaled is rescaled to the band chosen for rescaling (usually, the most numerous)
                        # adding the weighted mean rescaling factor (ff[2])
                        lightrescy[pp]=lightrescy[pp]+ff[2]

                        # the error on the rescaled magnitude is given by the square root of the squared sum of
                        # magnitude error and the error on the weighted mean rescaling factor (ff[3])
                        lightrescyerr[pp]=np.sqrt((lightrescyerr[pp]**2)+(ff[3]**2))
                    else:
                        # Otherwise, if not all of the 3 conditions hold, the point is not rescaled
                        lightrescy[pp]=lightrescy[pp]

        # The plot of the rescaled dataframe
        figresc = px.scatter(
                x=np.log10(lightrescx), # the time is set to log10(time) only in the plot frame
                y=lightrescy,
                error_y=lightrescyerr,
                color=lightresccolor,
                )

        font_dict=dict(family='arial',
                    size=18,
                    color='black'
                    )

        figresc['layout']['yaxis']['autorange'] = 'reversed'
        figresc.update_yaxes(title_text="<b>Magnitude<b>",
                        title_font_color='black',
                        title_font_size=18,
                        showline=True,
                        showticklabels=True,
                        showgrid=False,
                        linecolor='black', 
                        linewidth=2.4, 
                        ticks='outside', 
                        tickfont=font_dict,
                        mirror='allticks', 
                        tickwidth=2.4, 
                        tickcolor='black',  
                        )

        figresc.update_xaxes(title_text="<b>log10 Time (s)<b>",
                        title_font_color='black',
                        title_font_size=18,
                        showline=True,
                        showticklabels=True,
                        showgrid=False,
                        linecolor='black',
                        linewidth=2.4,
                        ticks='outside',
                        tickfont=font_dict,
                        mirror='allticks',
                        tickwidth=2.4,
                        tickcolor='black',
                        )

        figresc.update_layout(title="GRB " + self.name + " rescaled",
                        title_font_size=25,
                        font=font_dict,
                        plot_bgcolor='white',  
                        width=960,
                        height=540,
                        margin=dict(l=40,r=40,t=50,b=40)
                        )

        figresc.show()

        # The definition of the rescaled dataframe
        # the list of values must be merged again in a new dataframe before exporting

        rescdataframe = pd.DataFrame()
        rescdataframe['time_sec'] = lightrescx
        rescdataframe['mag_rescto_'+str(filterforrescaling)] = lightrescy
        rescdataframe['mag_err'] = lightrescyerr
        rescdataframe['band_og'] = self.band_og 
        rescdataframe['system'] = lightrescsystem
        rescdataframe['telescope'] = lightresctelescope
        rescdataframe['extcorr'] = lightrescextcorr
        rescdataframe['source'] = lightrescsource

        # The option for exporting the rescaled magnitudes as a dataframe
        rescdataframe.to_csv(os.path.join(save_rescaled_in+'/'+str(self.name)+'_rescaled_to_'+str(filterforrescaling)+'.txt'),sep=' ',index=False)

        return rescdataframe

major, *__ = sys.version_info # this command checks the Python version installed locally
readfile_kwargs = {"encoding": "utf-8"} if major >= 3 else {} # this option specifies the enconding of imported files in Python
                                                              # the encoding is utf-8 for Python versions superior to 3. 
                                                              # otherwise it is left free to the code

def _readfile(path): # function for basic importation of text files, using the options defined in lines 1043,1044   
    with open(path, **readfile_kwargs) as fp:
        contents = fp.read()
    return contents

# re.compile(): compile the regular expression specified by parenthesis to make it match
version_regex = re.compile('__version__ = "(.*?)"') # 
contents = _readfile(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "__init__.py"
    )
) # this command reads __init__.py that gives the basic functions for the package, namely get_dir, set_dir
__version__ = version_regex.findall(contents)[0]

__directory__ = get_dir() 