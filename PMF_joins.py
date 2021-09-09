import math
import matplotlib.pyplot as plt
import pandas as pd
import random


# go through the PMF, ensure area = 1
def normalise(pmf):

    total_prob = 0

    if(len(pmf) > 0):

        n_pmin = pmf[0]['price']
        n_pmax = pmf[-1]['price']

        cum_prob = 0

        # go through current PMF, calculate the total PMF
        for i in range(int(n_pmin), n_pmax + 1):

            try:
                total_prob += pmf[i - n_pmin]['prob']
            except:
                total_prob += 0

        # go through PMF range of prices, adjust probability as each price
        # so total area under curve = 1
        for i in range(int(n_pmin), n_pmax + 1):

            try:
                if(total_prob > 0):
                    pmf[i - n_pmin]['prob'] /= total_prob
                    cum_prob += pmf[i - n_pmin]['prob']
                    pmf[i - n_pmin]['cum_prob'] = cum_prob
            except:
                total_prob += 0




# calculate cumulative distribution function (CDF) look-up table (LUT)

# At the heart of PRZI is a function P(p), i.e., eq 8 in technical note
# -- in calc_LUT.py this is the function from line 59 to line 75
# Following the calculation of caligrahpic price for each price (eq 8)
# we convert these prices (cal_p) into a usable PMF, and then CDF via
# summing and normalising (eq 11) to give probs
def calc_cdf_lut(strat, t0, tang_m, dirn, pmin, pmax):

    # set parameter values and calculate CDF LUT
    # dirn is direction: -1 for buy, +1 for sell

    # the threshold function used to clip
    def threshold(theta0, x):
        t = max(-1*theta0, min(theta0, x))
        return t

    epsilon = 0.000001 #used to catch DIV0 errors
    verbose = False

    if (strat > 1.0) or (strat < -1.0):
        # out of range
        sys.exit('FAIL: PRZI.getorder() self.strat out of range\n')

    if (dirn != 1.0) and (dirn != -1.0):
        # out of range
        sys.exit('FAIL: PRZI.calc_cdf() bad dirn\n')

    if pmax < pmin:
        # screwed
        sys.exit('FAIL: pmax < pmin\n')

    dxs = dirn * strat

    if verbose:
        print('calc_cdf_lut: dirn=%d dxs=%d pmin=%d pmax=%d\n' % (dirn, dxs, pmin, pmax))

    p_range = float(pmax - pmin)
    if p_range < 1:
        # special case: the SHVR-style strategy has shaved all the way to the limit price
        # the lower and upper bounds on the interval are adjacent prices;
        # so cdf is simply the lower price with probability 1

        cdf=[]
        cdf.append({'price': pmin, 'prob': 1, 'cal_p': 0, 'cal_sum': 0, 'cum_prob': 1})

        if verbose:
            print('\n\ncdf:', cdf)

        return {'strat': strat, 'pmin': pmin, 'pmax': pmax, 'dirn': dirn, 'cdf_lut': cdf}

    c = threshold(t0, tang_m * math.tan(math.pi * (strat + 0.5)))

    # catch div0 errors here
    if abs(c) < epsilon:
        if c > 0:
            c = epsilon
        else:
            c = -epsilon

    e2cm1 = math.exp(c) - 1

    # calculate the discrete calligraphic-P function over interval [pmin, pmax]
    # (i.e., this is Equation 8 in the PRZI Technical Note)
    calp_interval = []
    cal_sum = 0
    for p in range(pmin, pmax + 1):
        p_r = (p - pmin) / (p_range)  # p_r in [0.0, 1.0]
        if strat == 0.0:
            # special case: this is just ZIC
            cal_p = 1 / (p_range + 1)
        elif strat > 0:
            cal_p = (math.exp(c * p_r) - 1.0) / e2cm1
        else:  # self.strat < 0
            cal_p = 1.0 - ((math.exp(c * p_r) - 1.0) / e2cm1)
        if cal_p < 0:
            cal_p = 0   # just in case
        calp_interval.append({'price':p, "cal_p":cal_p})
        cal_sum += cal_p

    if cal_sum <= 0:
        print('calp_interval:', calp_interval)
        print('pmin=%f, pmax=%f, cal_sum=%f' % (pmin, pmax, cal_sum))

    cdf = []
    cum_prob = 0
    # now go thru interval summing and normalizing to give the CDF
    # this is equation 12
    for p in range(pmin, pmax + 1):
        price = calp_interval[p-pmin]['price']
        cal_p = calp_interval[p-pmin]['cal_p']
        prob = cal_p / cal_sum                                         # this is the PMF
        cum_prob += prob
        cdf.append({'price': p, 'prob': prob, 'cal_p': cal_p, 'cal_sum': cal_sum, 'cum_prob': cum_prob})

    if verbose:
        print('\n\ncdf:', cdf)
        print('\n\n\n\n')

    verbose = False

    return {'strat':strat, 'dirn':dirn, 'pmin':pmin, 'pmax':pmax, 'cdf_lut':cdf}


# create a uniform PMF where area underneath curve = 0 (this covers edge cases)
# in some joins where resultant PMF has 0 probs for all prices
def null_probs(uni_pmf):

    pmin = uni_pmf['pmin']
    pmax = uni_pmf['pmax']

    for p in range(pmin, pmax + 1):

        uni_pmf['cdf_lut'][p - pmin]['prob'] = 0


# difference join composition mechanism
def D_join(pmf_orig_1, pmf_orig_2):

    pmf_orig_1 = pmf_orig_1.get_pmf()
    pmf_orig_2 = pmf_orig_2.get_pmf()

    # subtracts the PMF of the right-hand daughter from the PMF of the left-hand daughter,
    # does something sensible with any negative values that result (e.g., clip at zero,
    # or rebase so the most extreme negative value becomes the zero-point),
    # and then re-normalises to ensure that the area under the curve is one, as required for a PMF.

    pmf1_pmin = pmf_orig_1['pmin']
    #print('pmf1_pmin = {}'.format(pmf1_pmin))

    pmf1_pmax = pmf_orig_1['pmax']
    #print('pmf1_pmax = {}'.format(pmf1_pmax))

    pmf2_pmin = pmf_orig_2['pmin']
    #print('pmf2_pmin = {}'.format(pmf2_pmin))

    pmf2_pmax = pmf_orig_2['pmax']
    #print('pmf2_pmax = {}'.format(pmf2_pmax))

    new_pmf = []
    strat = pmf_orig_1['strat']
    dirn = pmf_orig_1['dirn']

    lowest_pmin = min(pmf1_pmin, pmf2_pmin)
    highest_pmax = max(pmf1_pmax, pmf2_pmax)
    # total range of prices
    r_total = (highest_pmax - lowest_pmin) + 1

    cum_prob = 0

    # go through all of the prices from both PMFs
    # for price p, subtract the right hand daughter's probability (if any for p, if not = 0) from the
    # ..left hand daughter's probability (if any for p, if not = 0)
    # we will end up with some negative probabilities no doubt, so pass the pmf through rebase function
    # to shift the entire PMF upwards

    for p in range(lowest_pmin, highest_pmax + 1):

        try:
            cal_p = pmf_orig_2['cdf_lut'][p - pmf2_pmin]['cal_p']
            cal_sum = pmf_orig_2['cdf_lut'][p - pmf2_pmin]['cal_sum']
            r_h_prob = pmf_orig_2['cdf_lut'][p - pmf2_pmin]['prob']
        except:
            r_h_prob = 0

        try:
            cal_p = pmf_orig_1['cdf_lut'][p - pmf1_pmin]['cal_p']
            cal_sum = pmf_orig_1['cdf_lut'][p - pmf1_pmin]['cal_sum']
            l_h_prob = pmf_orig_1['cdf_lut'][p - pmf1_pmin]['prob']
        except:
            l_h_prob = 0

        diff = l_h_prob - r_h_prob
        cum_prob += diff

        new_pmf.append({'price': p, 'prob': diff, 'cal_p': cal_p, 'cal_sum': cal_sum, 'cum_prob': cum_prob})

    # sometimes an join results in a PMF with an area of 0, hence set this to a uniform
    # PMF with 0 all the way across, and then normalise after
    if(len(new_pmf) == 0):
        new_pmf = calc_cdf_lut(0, 100, 4, dirn, lowest_pmin, highest_pmax)
        null_probs(new_pmf)

    # since now we will have negative probabilties, we must rebase the PMF
    rebase(new_pmf)
    normalise(new_pmf)

    return {'strat':strat, 'dirn':dirn, 'pmin':lowest_pmin, 'pmax':highest_pmax, 'cdf_lut':new_pmf}


def rebase(pmf):

    # rebasing the pmf requires finding the most-negative probability, and
    # making that the zero-point, i.e., making this most-negative probability positive,
    # and then adding that to every proability - this will in essence shift the PMF up

    r_pmin = pmf[0]['price']
    r_pmax = pmf[-1]['price']

    cum_prob = 0

    df = pd.DataFrame(pmf)
    lowest_prob = df.min()['prob']
    # make lowest_prob +ve
    lowest_prob = abs(lowest_prob)

    for p in range(r_pmin, r_pmax + 1):

        pmf[p - r_pmin]['prob'] += lowest_prob

        cum_prob += pmf[p - r_pmin]['prob']
        pmf[p - r_pmin]['cum_prob'] = cum_prob

    #df = pd.DataFrame(pmf)
    #lowest_prob = df.min()['prob']

# everything that is passed into this function will be a cal_p function already
# i.e., no need to convert constants to cal_p
# the in-series composition mechanism
def S_join(pmf_1, pmf_2):


    # get weightings on branches
    weight_left = pmf_1.get_weight()
    weight_right = pmf_2.get_weight()

    pmf_1 = pmf_1.get_pmf()
    pmf_2 = pmf_2.get_pmf()

    pmf1_pmin = pmf_1['pmin']
    #print('pmf1_pmin = {}'.format(pmf1_pmin))

    pmf1_pmax = pmf_1['pmax']
    #print('pmf1_pmax = {}'.format(pmf1_pmax))

    pmf2_pmin = pmf_2['pmin']
    #print('pmf2_pmin = {}'.format(pmf2_pmin))

    pmf2_pmax = pmf_2['pmax']
    #print('pmf2_pmax = {}'.format(pmf2_pmax))

    cum_prob = 0
    strat = pmf_1['strat']
    dirn = pmf_1['dirn']

    lowest_pmin = min(pmf1_pmin, pmf2_pmin)
    highest_pmax = max(pmf1_pmax, pmf2_pmax)
    r_total = (highest_pmax - lowest_pmin) + 1

    weight_left = weight_left / (weight_left + weight_right)
    weight_right = 1 - weight_left

    # the total number of price levels assigned to each PMF
    left_total = int(weight_left * r_total)
    right_total = r_total - left_total

    cal_sum1 = pmf_1['cdf_lut'][-1]['cal_sum']
    cal_sum2 = pmf_2['cdf_lut'][-1]['cal_sum']

    new_pmf = []
    cum_prob = 0

    # first segment of discretised prices
    for p in range(lowest_pmin, highest_pmax + 1):

        price = p
        # flag used for testing purposes
        found = 0

        if((p-lowest_pmin) <= left_total):

            try:
                if(pmf_1['cdf_lut'][p - pmf1_pmin]['price'] == price):
                    cal_p = pmf_1['cdf_lut'][p - pmf1_pmin]['cal_p']
                    prob = pmf_1['cdf_lut'][p - pmf1_pmin]['prob']
                    cum_prob += prob
                    new_pmf.append({'price': p, 'prob': prob, 'cal_p': cal_p, 'cal_sum': cal_sum1, 'cum_prob': cum_prob})
                    found = 1
            except:
                found = 0

        else:

            try:
                if(pmf_2['cdf_lut'][p - pmf2_pmin]['price'] == price):
                    cal_p = pmf_2['cdf_lut'][p - pmf2_pmin]['cal_p']
                    prob = pmf_2['cdf_lut'][p - pmf2_pmin]['prob']
                    cum_prob += prob
                    new_pmf.append({'price': p, 'prob': prob, 'cal_p': cal_p, 'cal_sum': cal_sum2, 'cum_prob': cum_prob})
                    found = 1
            except:
                found = 0

    normalise(new_pmf)

    # sometimes an join results in a PMF with an area of 0, hence set this to a uniform
    # PMF with 0 all the way across, and then normalise after
    if(len(new_pmf) == 0):
        new_pmf = calc_cdf_lut(0, 100, 4, dirn, lowest_pmin, highest_pmax)
        null_probs(new_pmf)

    return {'strat':strat, 'dirn':dirn, 'pmin':lowest_pmin, 'pmax':highest_pmax, 'cdf_lut':new_pmf}

def get_calp(pmf_1, price):

    pmf1_pmin = pmf_1['pmin']
    pmf1_pmax = pmf_1['pmax']

    if(pmf1_pmin > price or pmf1_pmax < price):
        return 0

    for i in range(pmf1_pmin, pmf1_pmax + 1):
        try:
            if(pmf_1['cdf_lut'][i - pmf1_pmin]['price'] == price):
                return pmf_1['cdf_lut'][i - pmf1_pmin]['cal_p']
        except:
            pass

# in parallel composition mechanism
def P_join(pmf_orig_1, pmf_orig_2):

    pmf_orig_1 = pmf_orig_1.get_pmf()
    pmf_orig_2 = pmf_orig_2.get_pmf()

    pmf1_pmin = pmf_orig_1['pmin']
    #print('pmf1_pmin = {}'.format(pmf1_pmin))

    pmf1_pmax = pmf_orig_1['pmax']
    #print('pmf1_pmax = {}'.format(pmf1_pmax))

    pmf2_pmin = pmf_orig_2['pmin']
    #print('pmf2_pmin = {}'.format(pmf2_pmin))

    pmf2_pmax = pmf_orig_2['pmax']
    #print('pmf2_pmax = {}'.format(pmf2_pmax))

    overlap = pmf1_pmax - (pmf2_pmin - pmf1_pmin)

    new_pmf = []
    cum_prob = 0
    cal_sum = pmf_orig_1['cdf_lut'][-1]['cal_sum'] + pmf_orig_2['cdf_lut'][-1]['cal_sum']
    strat = pmf_orig_1['strat']
    dirn = pmf_orig_1['dirn']

    lowest_pmin = min(pmf1_pmin, pmf2_pmin)
    highest_pmax = max(pmf1_pmax, pmf2_pmax)

    cum_prob = 0

    # go through all possible prices
    for p in range(lowest_pmin, highest_pmax + 1):

        price = p

        cal_p1 = get_calp(pmf_orig_1, p)
        cal_p2 = get_calp(pmf_orig_2, p)

        # get the total price, divide by cal_sum, and append to new_pmf
        cal_p_composite = cal_p1 + cal_p2
        if(cal_sum > 0):
            prob = cal_p_composite / cal_sum
        else:
            prob = 1
        cum_prob += prob
        new_pmf.append({'price': p, 'prob': prob, 'cal_p': cal_p_composite, 'cal_sum': cal_sum, 'cum_prob': cum_prob})

    normalise(new_pmf)

    return {'strat': strat, 'pmin': lowest_pmin, 'pmax': highest_pmax, 'dirn': dirn, 'cdf_lut': new_pmf}
