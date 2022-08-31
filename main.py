import random
from datetime import date
import yfinance as yf
import scipy.fft
import numpy as np
import matplotlib.pyplot as plotter
from scipy.fft import fft, ifft, fftfreq
import pandas as pd
import matplotlib.pyplot as plt
import scipy.fftpack
from scipy.fft import rfft, rfftfreq, irfft
from datetime import date
from scipy import signal
import math
import cmath
from collections import Counter
import statistics as stat


def ploting_regular_functions_abs_val(x, y, size):
    plt.figure(num=-10000.0, dpi=size)
    plt.plot(x, np.abs(y), 'k')
    plt.show()


def ploting_stem_functions_abs_val(x, y, size):
    plt.figure(num=-100.0, dpi=size)
    plt.stem(x, np.abs(y), use_line_collection=True)
    plt.show()


def score_of_stock_based_on_recommendations(company_tag):
    dictionary = {}
    i = 0
    while (i < company_tag.recommendations["To Grade"].count()):
        rec = company_tag.recommendations["To Grade"][i]
        if (dictionary.get(rec) == None):
            dictionary[rec] = 1
        else:
            dictionary[rec] += 1
        i += 1
    score = 0
    for rec in dictionary:
        number_of_item_recorded = dictionary[rec]
        if (rec == "Hold" or rec == "Positive"):
            score += number_of_item_recorded * 3
        if (rec == "Buy" or rec == "Long-Term Buy"):
            score += number_of_item_recorded * 10
        if (
                rec == "Overweight" or rec == "Outperform" or rec == "Strong Buy" or rec == "Market Outperform" or rec == "Sector Outperform"):
            score += number_of_item_recorded * 12
        if (
                rec == "Neutral" or rec == "Perform" or rec == "Market Perform" or rec == "Equal-weight" or rec == "Equal-Weight"):
            score += number_of_item_recorded * 0
        if (rec == "Sell"):
            score += number_of_item_recorded * -10
    average_score = score / company_tag.recommendations["To Grade"].count()
    print("Based on recommendations of analysts - the total score is:", average_score)
    print(dictionary)


def DFT(y):
    """
      Function to calculate the
      discrete Fourier Transform
      of a 1D real-valued signal x
      """

    N = len(y)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)

    yf = np.dot(e, y)
    return yf


def IDFT(yf):
    """
      Function to calculate the
      discrete Fourier Transform
      of a 1D real-valued signal x
      """

    N = len(yf)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(2j * np.pi * k * n / N)

    y = np.dot(e, yf) / N

    return y


def print_whole_list(list):
    i = 0
    while i < len(list):
        print(list[i])
        i += 1


def PSD(func):
    new_func = func * np.conj(func)
    return new_func


def zeros_builder_based_on_samples(x, func):
    i = 0
    zeros = 1
    while (i < len(func)):
        zeros *= (x - i)
        i += 1
    return zeros


def poles_builder_based_on_samples(x, func, value_index):
    i = 0
    poles = 1
    while (i < len(func)):
        if (i != value_index):
            poles *= (value_index - i)
        i += 1
    return poles


def building_polinom_for_given_func(x, func, zeros):
    i = 0
    polinom = 0
    while (i < len(func)):
        polinom += func[i] * (zeros / (x - i)) / (poles_builder_based_on_samples(x, func, i))
        i += 1
    return polinom


def FWHM(frq, absY_half):
    HM = (max(absY_half) - absY_half[len(absY_half) - 1]) / 2
    index_of_maximal_spectrum_val = absY_half.tolist().index(max(absY_half))
    print(f"The index of maximal value is: {index_of_maximal_spectrum_val}")
    i = 0
    entry1 = 0
    entry2 = 0
    while i < len(absY_half):

        if absY_half[i] >= HM and i <= index_of_maximal_spectrum_val and not (entry1):
            print(
                f"\nFRQ1 -> the position of y2, x2 in calc of the linear func will be {i}, {absY_half[i]},{HM}  and y1, x1 will be {i - 1} accordingly \n")
            m = (absY_half[i] - absY_half[i - 1]) / (frq[i] - frq[i - 1])
            b = absY_half[i] - m * frq[i]
            frq_1 = (HM - b) / m

            entry1 = 1

        if absY_half[i] <= HM and i >= index_of_maximal_spectrum_val:
            print(
                f"\nFRQ2 -> the position of y2, x2 in calc of the linear func will be {i} and y1, x1 will be {i - 1} accordingly \n")
            m = (absY_half[i] - absY_half[i - 1]) / (frq[i] - frq[i - 1])
            b = absY_half[i] - m * frq[i]
            # frq_2 = (HM - b) / m
            frq_2 = frq[i - 1]  #### THIS IS MOST CERTAINLY NOT CORRECT - WE ASSUME THAT THE HM IS WHERE THE MAXIMUM IS
            break

        i += 1
    delta_f = frq_2 - frq_1
    return delta_f


def Guessing_Real_Freq(fft_func):
    new_fft_func = fft_func.copy()
    i = 0
    while i < len(new_fft_func):
        i += 1
    return new_fft_func


# ########################################

# Fs = 20  # SAMPLE FREQUENCY!
# Ts = 1/50 # what will be the sampling rate when considering THE FUCKING FREQUENCIES OF THE SYSTEM.
# #more than twice the biggest freq - since the spectrom is devided into two and needs to include the biggest freq.
# t = np.arange(0,5,Ts) # time vector
#
# print(Fs/Ts)
# ff1 = 0.0231   # frequency of the signal and not OMEGA
# a1 = 1
# ff2 = 0.0523
# a2 = 1
# ff3 = 0.0056
# a3 = 1
# ff4 = 0.0079
# a4 = 1
# ff5 = 0.0133
# a5 = 1
# dc = 17.3293
#
# y = dc + a1*np.sin(2*np.pi*ff1*t) + a2*np.sin(2*np.pi*ff2*t) #+ a3*np.sin(2*np.pi*ff3*t) + a4*np.sin(2*np.pi*ff4*t) + a5*np.sin(2*np.pi*ff5*t)
#
# y_mean = sum(y)/len(y)
# print (f"The mean of all values of the given function PLUS zeros at the end is: {y_mean}")
# y = y - y_mean
#
# # y = np.sinc(np.exp(1j*t))
#
# #y=0
# #i=0
# # while (i<1):
# #       f = random.random()/2                  #creating the function
# #       a = random.randint(1,10)
# #       dc = random.randint(1,10)
# #       y += dc + a * np.sin(2*np.pi*f*t)
# #       i+=1
#
# plt.plot(t,y)
#
# #print(f, a, dc)
# # plt.show()
#
# #########################          ADDING WHITE NOISE
# # mean = 0
# # std = 0
# # num_samples = int(Fs/Ts)
# # # y += np.random.normal(mean, std, size=num_samples)
# # plt.plot(t,y)
# plt.show()
#
# ##########
# multiplications_number_concat = 1
# y_concat = y.copy()
# y = np.hstack([y_concat] * multiplications_number_concat)
#
# multiplications_number_zeros = 1                                                                # for better characterization in fft
#
# number_of_zeros = (multiplications_number_zeros-1)*len(y)
#
# i=0
# while i < number_of_zeros:                                                                #inserting zeros at the end of y_high function vector to maitain a better freq picture.
#       y = np.append(y, 0)
#       i= i+1
#
# mult_num = multiplications_number_zeros*multiplications_number_concat
#
# ##########
# n = len(y)                                      # length of the signal we created = Fs/Ts
#
# k = np.arange(n)                                #basically, an array the length of the signal we generated but it goes like: 0,1,2,3,... not acording to frequencies
# T = n*Ts                                        # we normalize the number of original - we divide the length of the signal after sampling it in the number of samples we wanted
# frq = k/T                                       # two sides frequency range
# frq = frq[:len(frq)//2]                         # one side frequency range
# Y = np.fft.fft(y)                               # dft and normalization
# Y_half = Y[:n // 2]
# a = np.angle(Y)
#
# a_half = a[:n // 2]
#
#
#
# y_new = 0
# The_length_I_want_to_see = Fs*2 ################ DO THE SAME IN THE TESTING SIGNAL TO CHECK IT OUT!!!!!!!
# steps = The_length_I_want_to_see/len(y)
# x = np.arange(0,The_length_I_want_to_see,steps) # BUILDING THE EXACT SAME VECTOR THAT WAS FOR THE ORIGINAL SIGNAL - t....
# i = 0
# max_val_abs_Y = max(abs(Y_half))
# while (i< len(frq)):
#       if (abs(Y_half[i]) > 0.001*max_val_abs_Y):            # whoever answers to the conditions will be represented in the recreated function
#
#             print(f"\nwe have frequency {i/(Fs*mult_num)} with the value of {abs(Y_half[i])} which is {abs(Y_half[i])/sum(abs(Y_half))*100}% from the total")
#
#             if i == 0 or i == (len(frq)-1):
#                   y_new += (abs(Y_half[i]) / n) * np.cos(2 * np.pi * (i / (Fs*mult_num)) * x + a[i])
#             else:
#                   y_new += 2 * (abs(Y_half[i]) / n) * np.cos(2 * np.pi * (i / (Fs*mult_num)) * x + a[i])    #I had to multiply by 2 because of the division in 2 of the frequency I BELIEVE - thats how it works.
#
#       i+=1
#
#
# plt.plot(x,y_new)
# plt.legend(["first one", "second one"])
# plt.show()
#
# rms = np.sqrt(np.mean(abs(Y_half)**2))
# print(f"the RMS value of all values is: {rms}")
# print(abs(Y_half))
# plt.stem(frq, abs(Y_half))                             # plotting the spectrum
# plt.xlabel('Freq (Hz)')
# plt.ylabel('|Y(freq)|')
# plt.title("Amplitude of fft")
# plt.show()
#
#
#
# plt.plot(frq, a_half)                                  #ploting the phase of the fft
# plt.xlabel('Freq (Hz)')
# plt.ylabel('Arg (frq)')
# plt.title("Phase of fft")
# plt.show()


###     REAL   CODE   BELOW:


###################################################

company_tag = yf.Ticker("GOOG")

today = date.today()
history_data = company_tag.history(period="1y",
                                   interval="1h")  # start="2022-03-14",  end= today)     # stock data assigned
# history_data = company_tag.history(start="2022-07-15",  end="2022-07-16", interval="1m")
history_data['Close'].plot(title=f"{company_tag}stock price ($)")
plt.show()

y_high = np.array([])  # creating array for the regular function

original_value_quantity = history_data["High"].count()  # number of values of the function
print(f"the number of values obtained: {original_value_quantity}")

i = 0
while (i < original_value_quantity):
    y_high = np.append(y_high, history_data["High"][
        i])  # inserting the values of the function into the y_high array to create the actual function
    i += 1

y_high = y_high[:int(original_value_quantity) - 1]
# y_high = y_high[int(original_value_quantity/2):]


# score_of_stock_based_on_recommendations(company_tag)

#############################################

# """POLINOM BUILT IN THE FUNCTIONS ABOVE - TRYING TO DO INTERPOLATION IN A SMART WAY"""
# x_polinom = np.arange(len(y_high))
# zeros = zeros_builder_based_on_samples(x_polinom,y_high)
# polinom = building_polinom_for_given_func(x_polinom,y_high,zeros)
#
# ploting_regular_functions_abs_val(x_polinom, polinom, 100)


#########################################


print(f" sum of y_high is: {sum(y_high)}")

"""an experiment of concatenating the same vector of values a few time to itself and then zero padding the result:"""

y_high_new_mean = sum(y_high) / len(y_high)
print(f"The mean of all values of the given function PLUS zeros at the end is: {y_high_new_mean}")
y_high_final = y_high.copy()
y_high_final = y_high_final - y_high_new_mean

print("HERE COMES YYYYYYYY - STEP")
i = 0
while i < len(y_high_final) - 1:
    y_high_final[i] = y_high[i + 1] - y_high[i]
    i += 1
y_high_final[i] = y_high_final[i - 1]

STEP = y_high_final.copy()
STEP_mean = sum(STEP) / len(STEP)
abs_STEP_mean = sum(abs(STEP)) / len(STEP)
print(f"\n\nSTEP MEAN IS: {STEP_mean} and the abs_STEP_MEAN IS: {abs_STEP_mean} \n\n")
x_step = np.arange(0, len(STEP))
plt.plot(x_step, STEP)
plt.title("STEP function")
plt.ylabel("Step size")
plt.xlabel("Step number")
plt.show()

print("HERE COMES THE SECOND YYYYYYY - SPEED")
i = 0
while i < len(y_high_final) - 1:
    y_high_final[i] = (abs(y_high_final[i + 1] - y_high_final[i])) ** 2         #This is energy calc
    # y_high_final[i] = y_high_final[i + 1] - y_high_final[i]
    i += 1
y_high_final[i] = y_high_final[i - 1]
SPEED = y_high_final.copy()
x_speed = np.arange(0, len(SPEED))
plt.plot(x_speed, SPEED)
plt.title("SPEED function")
plt.ylabel("SPEED size")
plt.xlabel("SPEED number")
plt.show()

print(
    f"\n\n\n\nTHE ENERGY OF THE STOCK IS: {np.dot(SPEED, SPEED)} and the new one is {np.dot(SPEED, SPEED) / abs_STEP_mean} \n\n\n\n\n\n\n")

print("HERE COMES THE THIRD YYYYYYY - ACCELERATION")
i = 0
while i < len(y_high_final) - 1:
    y_high_final[i] = y_high_final[i + 1] - y_high_final[i]
    i += 1
y_high_final[i] = y_high_final[i - 1]
ACCELERATION = y_high_final.copy()
x_acc = np.arange(0, len(ACCELERATION))
plt.plot(x_acc, ACCELERATION)
plt.title("ACCELERATION function")
plt.ylabel("ACCELERATION size")
plt.xlabel("ACCELERATION number")
plt.show()

#
# print("HERE COMES THE THIRD YYYYYYY - CHANGE OF ACCELERATION")
# i=0
# while i< len(y_high_final)-1:
#       y_high_final[i] = y_high_final[i+1] - y_high_final[i]
#       i+=1
# y_high_final[i] = y_high_final[i-1]
# D_ACCELERATION = y_high_final.copy()
# x_d_acc = np.arange(0,len(D_ACCELERATION))
# plt.plot(x_d_acc,D_ACCELERATION)
# plt.title("CHANGE OF ACCELERATION function")
# plt.ylabel("CHANGE OF ACCELERATION size")
# plt.xlabel("CHANGE OF ACCELERATION number")
# plt.show()

######################
# CHANGE_OF_E_OVER_TIME = np.dot(SPEED, y_high_final)
# CHANGE_OF_E_OVER_TIME = SPEED*y_high_final
# print(f"\n\n\n\nPLEASE WELCOME THE CANGE OF ENERGY!! = = = = = =\n {CHANGE_OF_E_OVER_TIME}\n\n\n\n")
# y_high_final = CHANGE_OF_E_OVER_TIME
######################


y_high_new_mean = sum(y_high_final) / len(y_high_final)  # FOR PRESENTING BETTER FRQ SHOW +
y_high_final = y_high_final - y_high_new_mean  # IT IS PRESENTED DIFFERENTLY IN THE SIGNAL PRESENTATION SINCE IT IS SUBSTRACTED BEFORE.

multi_concat = 1
number_of_concat = multi_concat
y_high_new = np.hstack([y_high_final] * number_of_concat)  # the actual concat - everytime it grows with 1
"""end of concat"""

multi_zero = 10
number_of_values_wanted = multi_zero * len(y_high_new)  # for better characterization in fft
number_of_zeros = number_of_values_wanted - len(y_high_new)

i = 0
while i < number_of_zeros:  # inserting zeros at the end of y_high function vector to maitain a better freq picture.
    y_high_new = np.append(y_high_new, 0)
    i = i + 1

multi = multi_zero * multi_concat

y_high_final = y_high_new.copy()
####################################################################


# the last test I did - quite successful - NEST TIME PRINTING WITH SIN INSTEAD OF EXP


Fs = float(len(y_high_final))  # range of sampling from 0
Ts = 1.0  # what will be the sampling rate when considering THE FUCKING FREQUENCIES OF THE SYSTEM.
t = np.arange(0, Fs, Ts)  # time vector

y = y_high_final.copy()
plt.plot(t, y)  #####################################################
# plt.show()


n = len(y)  # length of the signal we created = Fs/Ts
k = np.arange(
    n)  # basically, an array the length of the signal we generated but it goes like: 0,1,2,3,... not acording to frequencies
T = n * Ts  # we normalize the number of original - we divide the length of the signal after sampling it in the number of samples we wanted
frq = k / T  # two sides frequency range
frq = frq[:len(frq) // 2]  # one side frequency range
Y = np.fft.fft(y)  # dft and normalization
Y_half = Y[:n // 2]
a = np.angle(Y)
a_half = a[:n // 2]

y_new = 0
The_length_I_want_to_see = Fs  ################ DO THE SAME IN THE TESTING SIGNAL TO CHECK IT OUT!!!!!!!
steps = The_length_I_want_to_see / len(y)
x = np.arange(0, The_length_I_want_to_see,
              steps)  # BUILDING THE EXACT SAME VECTOR THAT WAS FOR THE ORIGINAL SIGNAL - t....
maximum = max(abs(Y_half))

TOP_REACHED = 0

i = 0
while (i < len(frq)):
    if (abs(Y_half[
                i]) > maximum / 40):  # whoever answers to the conditions will be represented in the recreated function
        # if (i < 16):
        print(f"we have frequency {i / (Fs * multi)} with the value of {abs(Y_half[i])}")
        if i == 0 or i == (len(frq) - 1):
            y_new += (abs(Y_half[i]) / n) * np.cos(2 * np.pi * (i / Fs) * x + a[i])
        else:
            y_new += 2 * (abs(Y_half[i]) / n) * np.cos(2 * np.pi * (i / Fs) * x + a[
                i])  # I had to multiply by 2 because of the division in 2 of the frequency I BELIEVE - thats how it works.
    i += 1

print_whole_list(abs(Y_half))

plt.plot(x, y_new)  #################################################
i = 0
diff = []
while (i < len(y_new)):
    diff.append(y[i] - y_new[i])
    i += 1

print(f"VAR of diff is: {stat.variance(diff)} and Estimation is: {stat.mean(diff)} ,  {stat.fmean(diff)}")

plt.plot(x, diff)  ###################################################
plt.legend(["first one", "second one", "difference"])
plt.show()

plt.plot(frq, abs(Y_half))  # plotting the spectrum
plt.xlabel('Freq (Hz)')
plt.ylabel('|Y(freq)|')
plt.title("Amplitude of fft")
plt.show()

delta_f = FWHM(frq, abs(Y_half))
print("delta f is", delta_f)
delta_tc = 1 / delta_f
print(f"\n\nTHE NUMBER OF SAMPLES RELEVANT TO IT IS {delta_tc}\n\n")

plt.plot(frq, a_half)  # ploting the phase of the fft
plt.xlabel('Freq (Hz)')
plt.ylabel('Arg (frq)')
plt.title("Phase of fft")
plt.show()

####################################################################
new_value_quantity = len(y_high_final)

"""using FFT here and not calculating by hand:"""

yf = np.fft.fft(y_high_final)  # Here we get good values for the
xf = np.fft.fftfreq(len(yf))

plt.plot(xf, yf)  #####AMAZING... SINC FUNC
plt.show()

"""plotting something"""
sampling_rate = 1.0
# calculate the frequency
N = len(yf)  # when fft is done we need to take the whole new signal range not the original
print("len of yf is:::::::::::::::", N)
n = np.arange(N)
T_sampling = 1.0
freq = n / T_sampling

############################################

"""correcting the FOURIER TRANSFORM from symmetrical to a good one with proper values"""
n_oneside = N // 2
# get the one side frequency
f_oneside = freq[:len(freq) // 2]

# normalize the amplitude
yf_func_oneside = yf[:n_oneside] / n_oneside

plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.stem(f_oneside, abs(yf_func_oneside), 'b', markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('DFT Amplitude |X(freq)|')

plt.show()

ploting_regular_functions_abs_val(f_oneside, abs(yf_func_oneside), 120)

no_hill_func = yf_func_oneside.copy()
number_of_hills = 0
number_of_hills_wanted = 4
number_of_freq_of_zero = 0
i = 1
while (i < len(no_hill_func)):
    if (number_of_hills == number_of_hills_wanted):
        no_hill_func[i] = 0
        number_of_freq_of_zero += 1
    elif (i == len(no_hill_func) - 1):
        no_hill_func[i] = 0
        break
    elif (abs(no_hill_func[i]) < abs(no_hill_func[i + 1]) and abs(no_hill_func[i]) < abs(no_hill_func[i - 1])):
        number_of_hills += 1

    i += 1
    print(f"the number of freqs included not zeros:{number_of_values_wanted - number_of_freq_of_zero}")
no_hill_func_fft = np.fft.ifft(
    no_hill_func)  ######THIS DOESNT WORK BECAUSE I NEED ONLY THE NUMBER OF SAMPLES AND I HAVE GOT A LOT MORE!
x_no_hill = np.arange(len(no_hill_func_fft))

# ploting_regular_functions_abs_val(x_no_hill, no_hill_func_fft, 170)
# ploting_stem_functions_abs_val(x_no_hill, no_hill_func_fft, 170)


"""SAMPLING THE FREQUENCY REPRESENTATION OF THE SIGNAL"""

# number of samples in freq
sn = 2
# sampling interval
fs = N // sn
f = np.arange(0, N, fs)
newly_yf_TEST2 = []
fs_tracker = 1
counter_of_samles_so_far = 0
i = 0
while i < len(yf):
    if i == fs_tracker:
        newly_yf_TEST2.append(yf[i])
        counter_of_samles_so_far += 1
        fs_tracker += fs
    i += 1
print(f"counter of samples so far:::::::{counter_of_samles_so_far}")
# print_whole_list(newly_yf_TEST2)

newly_y_TEST2 = ifft(newly_yf_TEST2)
newly_x_TEST2 = np.arange(len(newly_y_TEST2))

# ploting_regular_functions_abs_val(newly_x_TEST2, newly_y_TEST2, 70)


"""HERE I CLEAR THE FREQUENCY FUNCTION FROM NOISE SUPPOSEDLY"""
"""recreating the sinusoidal function to recommend on buying or selling"""
i = 0  # When concating - clears all but spikey freqs
sum = 0
number_of_frequencies_wanted = 1000
ratio_between_neighbors = 1
yf_clean = yf_func_oneside.copy()
is_it_first_freq = 0
while i < yf_func_oneside.size:

    if (sum == number_of_frequencies_wanted):
        yf_clean[i] = 0
        i += 1
        continue
    if (i == 0 or i == yf_func_oneside.size - 1):
        yf_clean[i] = 0
        i += 1
        continue
    elif ((abs(yf_func_oneside[i]) > ratio_between_neighbors * abs(yf_func_oneside[i - 1])) and (
            abs(yf_func_oneside[i]) > ratio_between_neighbors * abs(yf_func_oneside[i + 1]))):
        sum += 1
        if (is_it_first_freq == 0):
            first_freq_size = abs(yf_func_oneside[i])
        is_it_first_freq = 1
        print(
            f"THE NUMBER IS {i} AND IT POWER IS {abs(yf_func_oneside[i])} AND THE RATIO TO FIRST FREQ IS: {first_freq_size / abs(yf_func_oneside[i])}")
        i += 1


    else:
        yf_clean[i] = 0
        i += 1

print(f"Number of spikes found, INCLUDING those who are neighbors : {sum}")

yf_clean_extra_zeros = yf_clean.copy()
# i = 0
# while i < yf_clean.size:
#       if yf_clean[yf_clean.size - 1 - i] == 0:
#             yf_clean_extra_zeros = np.delete(yf_clean_extra_zeros, yf_clean.size - 1 - i)
#       else:
#             break
#       i += 1

# print(np.abs(yf_clean_extra_zeros))


# i = 0
# final_func_TEST = []
# while (i<len(y_recreation)-1):                                                #HERE I TRY TO PAD THE FREQ WITH ZEROS INBETWEEN
#       if i%2 == 0:
#             final_func_TEST.append(yf[i])
#       else:
#             final_func_TEST.append(0)
#       i+=1


# i = 0
# l = 0
# old_freq = 0
# final_function_output = 0
# length_of_output_wanted = 1*len(yf)
# x_of_final = np.arange(0, length_of_output_wanted,1)              #the plotted range - WE WANT THE FUTURE!       # 1/T_sampling gives us way better percission
# print(len(yf_func_oneside))
# while i < len(yf_func_oneside):
#       if (abs(yf_func_oneside[i]) > 0):
#             final_function_output += 2*(abs(yf_func_oneside[i])) * np.exp(2j*np.pi*i* x_of_final/length_of_output_wanted)
#
#             print(f"the {l}th is {i}")
#             l += 1
#       i+=1
#
#
#
#
# plt.plot(x_of_final, final_function_output)
# plt.show()
#############################################################################3

n = len(yf)  # length of the signal we created
k = np.arange(
    n)  # basically, an array the length of the signal we generated but it goes like: 0,1,2,3,... not acording to frequencies
T = n / n  # we normalize the number of original - we divide the length of the signal after sampling it in the number of samples we wanted
frq = k / T  # two sides frequency range
frq = frq[:len(frq) // 2]  # one side frequency range
Y_half = yf / n  # dft and normalization
Y_half = Y_half[:n // 2]

plt.plot(frq, abs(Y_half))  # plotting the spectrum
plt.xlabel('Freq (Hz)')
plt.ylabel('|Y(freq)|')
plt.title('THE FUCKING GREATEST FUCKING GRAPH EVER!!!!!')
plt.show()

y = 0
length_wanted = n * 3
x = np.arange(0, length_wanted, 1)
i = 0
while (i < len(frq)):
    if (abs(Y_half[i]) > 0):  # whoever answers to the conditions will be represented in the recreated function
        print(f"we have {i} with power of {abs(Y_half[i])}")
        y += 2 * (abs(Y_half[i])) * np.exp(2j * np.pi * i * x / (
                    n / 2))  # I had to multiply by 2 because of the division in 2 of the frequency I BELIEVE - thats how it works.
    i += 1

plt.plot(x, y)
plt.show()
