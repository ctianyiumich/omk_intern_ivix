import rqdatac as rq
from dotenv import load_dotenv
import os

import pandas as pd
import numpy as np
import datetime
from scipy.integrate import trapezoid as trap

from matplotlib import pyplot as plt

start_time = datetime.datetime.now()

# Initialize rqdatac, source of financial data
base_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = base_dir+'/VIX_DATA'
load_dotenv()
token_path = os.getenv("RQDATAC2_CONF")
os.environ['RQDATAC2_CONF'] = token_path
rq.init()
print(f'{datetime.datetime.now()-start_time}: Environment initialized')

# Functions
def days_year(year):
    """
    Returns total number of all trading days of the current year.
    Rqdata API access required.

    PARAMETERS:
        year: str/int, 4-digit number if year (e.g. 2024)
    
    RETURN:
        annus: int, number of trading days of the current year # "Annus" is the Latin word for "year".
    """
    first_day = f'{year}-01-01'
    last_day = f'{year}-12-31'
    anuus = len(rq.get_trading_dates(first_day, last_day))
    return anuus

def datestr_long2short(datestr):
    """
    Reformates string of date from the long format (e.g. "2024-09-30") to the short one (e.g. "2409") for maturity calculation.

    PARAMETERS:
        datestr: str, string of date in format of "%Y-%m-%d"

    RETURN:
        yymm: str, string of date in format of "%y%m"
    """
    ymdlist = datestr.split('-')
    yyyy = ymdlist[0]
    mm = ymdlist[1]
    yymm = yyyy[2:]+mm
    return yymm

def get_rf_list(trade_date, nrow):
    """
    Returns a list of risk-free interest rate of the day.
    Rqdata API access required.

    PARAMETERS:
        trade_date: str, trading date
        nrow: int, length of output list
    
    RETURN:
        foenora: list, list of interest rate of the day # "Foenora" is the Latin word for "increments/interests".
    """
    r_f_anual = rq.get_yield_curve(start_date=trade_date, end_date=trade_date)['1Y']
    r_f_value = r_f_anual.values[0]
    foenora = [r_f_value]*nrow
    return foenora

def s_cls_list(underlying_id, trade_date):
    """
    Return a list of underlying assert prices (close) on the given trade date.
    Rqdata API access required.

    PARAMETERS:
        underlying_id: str, code of the underlying asset/contract
        trade_date: str, trading date formatted in "%Y-%m-%d"
    
    RETURN:
        pretia: list, list of close prices # "Pretia" is the Latin word for "prices".
    """
    s_info = rq.get_price(underlying_id, start_date=trade_date, end_date=trade_date, frequency='1m')
    pretia = s_info.close.values.tolist()
    return pretia

def decompose_datetime(dt):
    """
    Separates three parts of a datetime object, consisting of hour, minute and second.
    """
    hour = int(datetime.datetime.strftime(dt, '%H'))
    minute = int(datetime.datetime.strftime(dt, '%M')) - 1
    second = int(datetime.datetime.strftime(dt, '%S'))
    return hour, minute, second

def minutes_to_maturity(trade_datetime_str, maturity_datetime_str):
    """
    Returns number of minutes from the trading time till the maturity (9:30 am on the maturity date).
    Rqdata API access required.
    """
    # Reformat both inputs
    trade_datetime = datetime.datetime.strptime(trade_datetime_str, "%Y-%m-%d %H:%M:%S")
    maturity_datetime = datetime.datetime.strptime(maturity_datetime_str, "%Y-%m-%d %H:%M:%S")
    # Retrieve time elements
    trade_hour, trade_minute, trade_second = decompose_datetime(trade_datetime)
    maturity_hour, maturity_minute, maturity_second = decompose_datetime(maturity_datetime)

    T_to_minus = trade_hour*60 + trade_minute + trade_second/60 # Time from 00:00 am to the trading time of the day.
    T_to_add = maturity_hour*60 + maturity_minute + maturity_second/60 # Time from 00:00 am to the maturity time of the day.
    day_minutes = 24*60
    n_days = len(rq.get_trading_dates(trade_datetime, maturity_datetime)) - 1 # number of trading days counted excluding maturity date.
    return n_days*day_minutes-T_to_minus+T_to_add

def generate_daily_data(underlying_id, trade_date, mat_gap):
    """
    Generates a dataframe of daily data, where all values for each field can be sorely determined by trading time.
    Rqdata API access required.

    PARAMETERS:
        underlying_id: str, code of the underlying asset/contract
        trade_date: str, trading date formatted in "%Y-%m-%d"
        mat_gap: numpy.timedelta64, gap in days between trading date and maturity date, normally set at 30
    
    RETURN:
        tempus_df: pandas.DataFrame, time-determined data for each trading date # "Tempus" is the Latin word for "time".
    """
    underlying_close = rq.get_price(underlying_id, start_date=trade_date, end_date=trade_date, frequency='1m').close.reset_index()
    trade_dates_list = underlying_close.datetime # all trading time in minutes given the date
    nrow = len(trade_dates_list) # number of minutes within a single trading date, normally equal to 240
    option_list = rq.options.get_contracts(underlying_id, trading_date=trade_date)

    near_mat_list = []
    next_mat_list = []
    for i in range(trade_dates_list.shape[0]):
        # Reformat strings og trade time to pandas.Timestamp
        trade_time = str(trade_dates_list[i]).split(' ')[1]
        trade_hour = int(trade_time.split(':')[0])
        trade_minute = int(trade_time.split(':')[1])
        trade_time = pd.Timestamp(trade_date) +\
            np.timedelta64(trade_hour, 'h') +\
            np.timedelta64(trade_minute, 'm')
        
        # list of maturity date for all related option contracts on each day
        T_list = []
        for option in option_list:
            maturity_longstr = rq.instruments(option).maturity_date
            T_list.append(pd.Timestamp(maturity_longstr) + np.timedelta64(9, 'h') + np.timedelta64(30, 'm'))
            
        
        maturities_list = list(set(T_list)) # list of unique maturities on each day
        maturities_list.sort()

        # find maturities for near term and next term based on mat_gap (normall 30 days)
        gap_bool = [t - trade_time < mat_gap for t in maturities_list]
        next_index = gap_bool.index(False)
        if next_index == 0:
            next_index = 1
        next_mat_list.append(maturities_list[next_index])
        near_index = next_index - 1
        near_mat_list.append(maturities_list[near_index])

    tempus_df = pd.DataFrame({
        'datetime': trade_dates_list,
        "r_f": get_rf_list(trade_date, nrow),
        'near_maturity_time': near_mat_list,
        'days_in_year': [days_year(int(trade_date.split('-')[0]))]*240,
        'T1_min': [minutes_to_maturity(str(trade_dates_list[i]), str(near_mat_list[i])) for i in range(240)],
        'next_maturity_time': next_mat_list,
        'T2_min': [minutes_to_maturity(str(trade_dates_list[i]), str(next_mat_list[i])) for i in range(240)],
        'close': s_cls_list(underlying_id, trade_date)
    })
    return tempus_df

def write_file(object_dir):
    """
    Creates an empty folder/file under the input directory if it doesn't exist.
    """
    if not os.path.exists(object_dir):
        os.makedirs(object_dir)

def write_daily_data(underlying_id, date_str, T_gap, inplace=False):
    """
    Writes daily data, generated by function generate_daily_data(), via function write_file()

    PARAMETERS:
        underlying_id: str, code of the underlying asset/contract
        date_str: str, trading_date
        T_gap: numpy.timedelta64, gap in days between trading date and maturity date, normally set at 30
        inplace: bool, set at False by default to use existing data as much as possible
    """
    date_dir = data_dir+f'/{underlying_id}/{date_str}'
    csv_dir = f"{date_dir}/{date_str}.csv"
    if inplace==False:
        if not os.path.exists(csv_dir):
            try:
                panel_df = generate_daily_data(underlying_id, date_str, T_gap)
                write_file(date_dir)
                panel_df.to_csv(csv_dir)
            except AttributeError:
                print('No data')
        else:
            print(f'Data of {date_str} already exist')
    else:
        try:
            panel_df = generate_daily_data(underlying_id, date_str, T_gap)
            print(panel_df)
            write_file(date_dir)
            panel_df.to_csv(csv_dir)
        except AttributeError:
            print('No data')

def read_daily_data(underlying_id, date_str):
    """
    Reads daily data from existing directory
    """
    try:
        csv_dir = data_dir+f'/{underlying_id}/{date_str}/{date_str}.csv'
        data_df = pd.read_csv(csv_dir, index_col='datetime')
        return data_df
    except FileNotFoundError:
        print(f'Data of {date_str} not found')

def get_option_prices(underlying, trade_date, trade_time, maturity):
    """
    Generates dataframe of contract data for each trading minute of the day.
    Rqdata API access required.

    PARAMETERS:
        underlying: str, code of the underlying asset/contract
        trade_date: str, trading date formatted in "%Y-%m-%d"
        trade_time: str, trading time formatted in "%H:%M:%S"
        maturity: str, maturity date formatted in "%y%m"
    
    RETURN:
        contracta_df: pandas.DataFrame, dataframe of contract data # "Contracta" is a Latin word for "contracts".
    """
    csv_dir = data_dir+f'/{underlying}/{trade_date}/{trade_time}options{maturity}.csv'
    
    # separate call and put options into two lists
    contracts_call_list = rq.options.get_contracts(underlying, option_type='C' ,maturity=maturity)
    contracts_put_list = rq.options.get_contracts(underlying, option_type='P' ,maturity=maturity)

    # add fields based on contracts
    contracts_call_df = pd.DataFrame({'strike_price': [inst.strike_price for inst in rq.instruments(contracts_call_list)],
                                    'order_book_id': [inst.order_book_id for inst in rq.instruments(contracts_call_list)]})
    contracts_put_df = pd.DataFrame({'strike_price': [inst.strike_price for inst in rq.instruments(contracts_put_list)],
                                    'order_book_id': [inst.order_book_id for inst in rq.instruments(contracts_put_list)]})
    contracts_call_df['option_type'] = ['C'] * contracts_call_df.shape[0]
    contracts_put_df['option_type'] = ['P'] * contracts_call_df.shape[0]
    contracts_call_df = contracts_call_df.reset_index()
    contracts_put_df = contracts_put_df.reset_index()

    # contracts_dict pairs call and puts under the same strike price
    if contracts_call_df.index.any() == contracts_put_df.index.any():
        contracts_dict = {
            'strike_price': contracts_call_df.strike_price,
            'call_obid': contracts_call_df.order_book_id,
            'put_obid': contracts_put_df.order_book_id,
        }
    else:
        print('Error: Strike prices do not match.')
    contracts_df = pd.DataFrame(contracts_dict)
    
    # get prices for calls and puts
    call_prices_df = rq.get_price(contracts_df.call_obid, start_date=trade_date, end_date=trade_date, frequency='1m', time_slice=(trade_time, trade_time)).reset_index().set_index('order_book_id').close.reset_index()
    put_prices_df = rq.get_price(contracts_df.put_obid, start_date=trade_date, end_date=trade_date, frequency='1m', time_slice=(trade_time, trade_time)).reset_index().set_index('order_book_id').close.reset_index()

    # assemble output dataframe
    contracta_df = contracts_df.merge(call_prices_df, left_on='call_obid', right_on='order_book_id')
    contracta_df = contracta_df.merge(put_prices_df, left_on='put_obid', right_on='order_book_id')
    contracta_df = pd.DataFrame(contracta_df, columns=['strike_price', 'call_obid', 'put_obid', 'close_x', 'close_y'])
    contracta_df.columns = ['strike_price', 'call_obid', 'put_obid', 'call_close', 'put_close']
    contracta_df = contracta_df.sort_values(by=['strike_price'])
    contracta_df['put_call_diff'] = abs(contracta_df.call_close - contracta_df.put_close)
    contracta_df['put_call_min'] = contracta_df.call_close.combine(contracta_df.put_close, min)
        
    contracta_df = contracta_df.set_index('strike_price')
    contracta_df.to_csv(csv_dir)

    return contracta_df

def read_option_prices(underlying, trade_date, trade_time, maturity, contracta_data_inplace):
    """
    Reads option data from existing directory
    """
    csv_dir = data_dir+f'/{underlying}/{trade_date}/{trade_time}options{maturity}.csv'
    if contracta_data_inplace == False:
        try:
            contractae_df = pd.read_csv(csv_dir, index_col='strike_price')
        except FileNotFoundError:
            contractae_df = get_option_prices(underlying, trade_date, trade_time, maturity)
    elif contracta_data_inplace == True:
        contractae_df = get_option_prices(underlying, trade_date, trade_time, maturity)
    return contractae_df

def get_call_status(k, s):
    """
    Separates out-of-money calls from the rest.
    """
    if s-k < 0:
        return 1
    else:
        return 0

def get_put_status(k, s):
    """
    Separates out-of-money puts from the rest.
    """
    if k-s <0:
        return 1
    else:
        return 0
"""
def call_k0_status(k0, kc):
    if kc>k0:
        return 1
    else:
        return 0

def put_k0_status(k0, kp):
    if kp<k0:
        return 1
    else:
        return 0
"""
def calculate_variance(underlying_id, trade_date, trade_time, maturity_date, T, S, R_f, read_option_inplace):
    """
    Calculates variance/sigma^2 based on given inputs

    PARAMETERES:
        underlying: str, code of the underlying asset/contract
        Calculates vairance/sigma^2 with given inputs
        trade_date: str, trading date formatted in "%Y-%m-%d"
        trade_time: str, trading time formatted in "%H:%M:%S"
        maturity_date: str, maturity date formatted in "%y%m"
        T: float, ratio of time till exploration to time in a year
        S: float/int, instantaneous price of underlying asset
        R_f: risk-free rate of the trading day
        read_option_inplace: bool, False to replace the old option data
    
    Returns:
        sigma_square: float, resulted vairance
    """
    contracti_df = read_option_prices(underlying_id, trade_date, trade_time, maturity_date, read_option_inplace)

    # Not saved as local data for flexibility of calculation
    contracti_df['pc_avg'] = 0.5*contracti_df.call_close + 0.5*contracti_df.put_close
    contracti_df['call_ofm'] = [get_call_status(k, S) for k in contracti_df.index]
    contracti_df['put_ofm'] = [get_put_status(k, S) for k in contracti_df.index]
    contracti_df['F'] = (contracti_df.call_close - contracti_df.put_close)*np.exp(R_f*T) + contracti_df.index

    call_atm_k = contracti_df.index.tolist()[contracti_df.call_ofm.values.tolist().index(1)]
    put_atm_k = contracti_df.index.tolist()[contracti_df.call_ofm.values.tolist().index(1)-1]
    diff_min = min(contracti_df.put_call_diff[call_atm_k], contracti_df.put_call_diff[put_atm_k])
    
    CP_diff_idxmin = contracti_df['put_call_diff'].idxmin()
    F_hat = contracti_df[contracti_df.put_call_diff==diff_min].F.values[0]
    try:
        K_0 = max(contracti_df[contracti_df.index<F_hat].index)
    except ValueError:
        K_0 = 2*contracti_df.index.values[0] - contracti_df.index.values[1]

    contracti_df['Q(K)'] = contracti_df.call_close * contracti_df.call_ofm \
                        + contracti_df.put_close * contracti_df.put_ofm
    if K_0 >= contracti_df.index.values[0]:
        contracti_df.loc[K_0,'Q(K)'] = contracti_df.pc_avg[K_0]
    
    second_term = (F_hat/K_0-1)**2/T

    q_k = pd.DataFrame({'strike': contracti_df.index, 'price':contracti_df['Q(K)']}).set_index('strike')
    q_k['value'] = q_k.price*np.exp(T*R_f)/(q_k.index**2)
    first_term = 2*trap(q_k.value, q_k.index)/T

    sigma_square = first_term - second_term
    return sigma_square

def calculate_vix(sigma_square1, sigma_square2, annus, N_T1, N_T2):
    """
    PARAMETERS:
        sigma_square1: float, variance of the near term 
        sigma_square2: float, variance of the next term
        annus: int, days in year
        N_T1: int, number of minutes of the near term
        N_T2: int, number of minutes of the next term

    RETURN:
        vix_ultima: float, ultimate result of vix calculation
    """
    N_30= 30*24*60
    T1 = N_T1/(annus*24*60)
    T2 = N_T2/(annus*24*60)
    coef_1 = (N_T2 - N_30)/(N_T2 - N_T1)
    coef_2 = (N_30 - N_T1)/(N_T2 - N_T1)

    vix_ultima = 100*np.sqrt(annus/30*(T1*sigma_square1*coef_1 + T2*sigma_square2*coef_2))
    return vix_ultima

def assemble_daily_vix(underlying_id, datestr, cal_var_inplace):
    """
    Please take it as an assembly line putting all gears together,
    which can be divided into three parts: variance for near term, variance for next term, ultimate vix.
    Returns a pandas dataframe of vix values for each trading minute.
    """
    sample_daily_df = read_daily_data(underlying_id, datestr)
    sample_daily_df['trade_date'] = [datetime.datetime.strptime(datestr.split(' ')[0], "%Y-%m-%d").strftime('%Y-%m-%d') for datestr in sample_daily_df.index]
    sample_daily_df['trade_time'] = [datetime.datetime.strptime(datestr.split(' ')[1], "%H:%M:%S").strftime('%H:%M') for datestr in sample_daily_df.index]
    sample_daily_df['near_yymm'] = [datetime.datetime.strptime(datestr.split(' ')[0], "%Y-%m-%d").strftime('%y%m') for datestr in sample_daily_df.near_maturity_time]
    sample_daily_df['next_yymm'] = [datetime.datetime.strptime(datestr.split(' ')[0], "%Y-%m-%d").strftime('%y%m') for datestr in sample_daily_df.next_maturity_time]
    sample_daily_df['T1'] = sample_daily_df.T1_min/(sample_daily_df.days_in_year*24*60)
    sample_daily_df['T2'] = sample_daily_df.T2_min/(sample_daily_df.days_in_year*24*60)
    print('Calculating near term variance:')
    u_list = []
    i=1
    for dt in sample_daily_df.index:
        nrow = sample_daily_df.shape[0]
        u_list.append(calculate_variance(underlying_id,
            sample_daily_df.loc[dt,:].trade_date,
            sample_daily_df.loc[dt,:].trade_time,
            sample_daily_df.loc[dt,:].near_yymm,
            sample_daily_df.loc[dt,:].T1,
            sample_daily_df.loc[dt,:].close,
            sample_daily_df.loc[dt,:].r_f,
            cal_var_inplace))
        print(f'{datestr} - near term vairance: {i} of {nrow} completed')
        i += 1
    sample_daily_df['near_v'] = u_list
    print('Calculating next term variance:')
    v_list = []
    i=1
    for dt in sample_daily_df.index:
        nrow = sample_daily_df.shape[0]
        v_list.append(calculate_variance(underlying_id,
            sample_daily_df.loc[dt,:].trade_date,
            sample_daily_df.loc[dt,:].trade_time,
            sample_daily_df.loc[dt,:].next_yymm,
            sample_daily_df.loc[dt,:].T2,
            sample_daily_df.loc[dt,:].close,
            sample_daily_df.loc[dt,:].r_f,
            cal_var_inplace))
        print(f'{datestr} - next term variance: {i} of {nrow} completed')
        i += 1
    sample_daily_df['next_v'] = v_list
    print('Calculating VIX:')
    vix_list = []
    i=1
    for dt in sample_daily_df.index:
        nrow = sample_daily_df.shape[0]
        vix_list.append(calculate_vix(
            sample_daily_df.loc[dt,:].near_v,
            sample_daily_df.loc[dt,:].next_v,
            sample_daily_df.loc[dt,:].days_in_year,
            sample_daily_df.loc[dt,:].T1_min,
            sample_daily_df.loc[dt,:].T2_min
        ))
        print(f'{datestr} - VIX: {i} of {nrow} completed')
        i += 1
    sample_daily_df['vix'] = vix_list
    return pd.DataFrame(sample_daily_df.vix)


def vix_time_data(underlying_id, trade_date_list, T_gap, tempus_data_inplace):
    """
    Saves daily data to the local direcotry, following the order of trade_date_list.
    """
    for i in range(len(trade_date_list)):
        n = len(trade_date_list)
        write_daily_data(underlying_id,trade_date_list[i], T_gap, inplace=tempus_data_inplace)
        print(f'{i+1} of {n} completed: {trade_date_list[i]}')

def vix_organize_data(underlying_id, trade_date_list, assemble_inplace):
    """
    Launches the function assemble_daily_vix(), following the order of trade_date_list.
    """
    vix_daily_list = []
    for date in trade_date_list:
        print(f'Calculating VIX on {date}:')
        try:
            vix_df = assemble_daily_vix(underlying_id, date, assemble_inplace)
            vix_daily_list.append(vix_df)
        except AttributeError:
            print(f'{date}: Insufficient Data')
    return vix_daily_list

def vix_concate(vix_daily_list, underlying_close):
    """
    Merge all daily dataframe by datetime axis.
    """
    s_series = underlying_close.close

    vix_longus_df = pd.concat(vix_daily_list)
    vix_longus_df = vix_longus_df.reset_index()
    vix_longus_df['t'] = pd.to_datetime(vix_longus_df.index)
    vix_longus_df['s'] = [s_series[timestr] for timestr in vix_longus_df.index]
    return vix_longus_df

def vix_reference1():
    referentia_filename = '1000INDX_ivix_20240429_20240701.csv'
    referentia_df = pd.read_csv(base_dir+'/'+referentia_filename) # Please doublecheck if reference data is saved under this directory
    return referentia_df.ivix


def vix_plot(vix_longus_df, underlying_id, trade_date_list, reference=False):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(vix_longus_df.index, vix_longus_df.s, color='red', label=f'{underlying_id}')

    if reference==True:
        vix_longus_df['vix_ref'] = vix_reference1()
        ax2.plot(vix_longus_df.index, vix_longus_df.vix_ref, color='green', label='vix_reference')
    else:
        pass
    
    ax2.plot(vix_longus_df.index, vix_longus_df.vix, color='blue', label='vix')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    n = len(trade_date_list)
    xtick = list(range(0, n, 240))
    xtick_labels = [vix_longus_df.datetime[i].split(' ')[0] for i in range(0, vix_longus_df.shape[0], 240)]
    
    ax1.set_xticks(list(range(0, vix_longus_df.shape[0], 240)), minor=False)
    ax1.set_xticklabels(xtick_labels, minor=False, rotation=30)

    ax1.grid()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc=0)
    ax1.set_title(f'iVIXP-{underlying_id}, {trade_date_list[0]}~{trade_date_list[-1]}\n Latest VIX: {round(vix_longus_df.vix.values.tolist()[-1], 4)}')

    print('Total time consumed: ', datetime.datetime.now()- start_time)
    plt.show()

def main():
    # Inputs (of your choice)
    #underlying_id = '588000.XSHG'
    underlying_id = '000852.XSHG'
    starting_date = '2024-06-01'
    ending_date = datetime.datetime.now().strftime('%Y-%m-%d')

    T_gap = np.timedelta64(30, 'D')
    time_data_inplace = False
    option_data_inplace = False
    
    # Let's Rock and  Roll
    # Find valid trading dates based on inputs
    trading_interval_list = rq.get_trading_dates(starting_date, ending_date)
    starting_date = trading_interval_list[0]
    ending_date = trading_interval_list[-1]

    try:
        get_rf_list(ending_date, 1)
    except TypeError:
        ending_date = trading_interval_list[-2] # If current interest rate is not yet available, move to the previous trading date

    r_f_anual = rq.get_yield_curve(start_date=starting_date, end_date=ending_date)['1Y']
    trade_idx = rq.get_price(underlying_id, start_date=starting_date, end_date=ending_date, frequency='1d').index.tolist()
    trade_date_list = [str(idx[1]).split(' ')[0] for idx in trade_idx]
    print(f'{datetime.datetime.now()-start_time}: Time data loaded')

    underlying_close = rq.get_price(underlying_id, start_date=starting_date, end_date=ending_date, frequency='1m').close.reset_index()
    underlying_close.datetime = [str(dt) for dt in underlying_close.datetime]
    underlying_close = underlying_close.set_index('datetime').drop('order_book_id', axis=1)

    print(f'{datetime.datetime.now()-start_time}: Prices of underlying asset loaded')

    vix_time_data(underlying_id, trade_date_list, T_gap, time_data_inplace)
    vix_daily_list = vix_organize_data(underlying_id, trade_date_list, option_data_inplace)
    vix_long_df = vix_concate(vix_daily_list, underlying_close)
    print(f'{datetime.datetime.now()-start_time}: Mission accomplished! Check plot')
    
    #reference_df = vix_reference1()
    vix_plot(vix_long_df, underlying_id, trade_date_list, reference=False)
    vix_long_df = vix_long_df.drop(['t', 's'], axis=1).set_index('datetime')
    vix_long_df.to_csv(data_dir+f'/iVIXP-{underlying_id}.csv')


if __name__ == '__main__':
    main()