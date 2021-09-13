# an Order/quote has a trader id, a type (buy/sell) price, quantity, timestamp, and unique i.d.
class Order:

        def __init__(self, tid, otype, price, qty, time, qid):
                self.tid = tid      # trader i.d.
                self.otype = otype  # order type
                self.price = price  # price
                self.qty = qty      # quantity
                self.time = time    # timestamp
                self.qid = qid      # quote i.d. (unique to each quote)

        def __str__(self):
                return '[%s %s P=%03d Q=%s T=%5.2f QID:%d]' % \
                       (self.tid, self.otype, self.price, self.qty, self.time, self.qid)

##################--Traders below here--#############
import random
import math
import timeit
import PMF_joins
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import sys

# Trader superclass
# all Traders have a trader id, bank balance, blotter, and list of orders to execute
class Trader:

    def __init__(self, ttype, tid, balance, time, opinion, uncertainty, lower_op_bound, upper_op_bound, start_opinion):
        self.ttype = ttype  # what type / strategy this trader is
        self.tid = tid  # trader unique ID code
        self.balance = balance  # money in the bank
        self.blotter = []  # record of trades executed
        self.orders = []  # customer orders currently being worked (fixed at 1)
        self.n_quotes = 0  # number of quotes live on LOB
        self.birthtime = time  # used when calculating age of a trader/strategy
        self.profitpertime = 0  # profit per unit time
        self.n_trades = 0  # how many trades has this trader done?
        self.lastquote = None  # record of what its last quote was

        self.opinion = opinion        # opinion between [0,1]
        self.uncertainty = uncertainty # uncertainty between [0, 2]

        self.lower_op_bound = lower_op_bound
        self.upper_op_bound = upper_op_bound
        self.lower_un_bound = 0
        self.upper_un_bound = 2

        self.start_opinion = start_opinion
        self.n_iter = 0

    def __str__(self):
        return '[TID %s type %s balance %s blotter %s orders %s n_trades %s profitpertime %s]' \
               % (self.tid, self.ttype, self.balance, self.blotter, self.orders, self.n_trades, self.profitpertime)

    def add_order(self, order, verbose):
        # in this version, trader has at most one order,
        # if allow more than one, this needs to be self.orders.append(order)
        if self.n_quotes > 0:
            # this trader has a live quote on the LOB, from a previous customer order
            # need response to signal cancellation/withdrawal of that quote
            response = 'LOB_Cancel'
        else:
            response = 'Proceed'
        self.orders = [order]
        if verbose:
            print('add_order < response=%s' % response)
        return response

    def del_order(self, order):
        # this is lazy: assumes each trader has only one customer order with quantity=1, so deleting sole order
        # CHANGE TO DELETE THE HEAD OF THE LIST AND KEEP THE TAIL
        self.orders = []

    def bookkeep(self, trade, order, verbose, time):

        outstr = ""
        for order in self.orders:
            outstr = outstr + str(order)

        self.blotter.append(trade)  # add trade record to trader's blotter
        # NB What follows is **LAZY** -- assumes all orders are quantity=1
        transactionprice = trade['price']
        if self.orders[0].otype == 'Bid':
            profit = self.orders[0].price - transactionprice
        else:
            profit = transactionprice - self.orders[0].price
        self.balance += profit
        self.n_trades += 1
        self.profitpertime = self.balance / (time - self.birthtime)

        if profit < 0:
            print(profit)
            print(trade)
            print(order)
            sys.exit()

        if verbose: print('%s profit=%d balance=%d profit/time=%d' % (outstr, profit, self.balance, self.profitpertime))
        self.del_order(order)  # delete the order

    # specify how trader responds to events in the market
    # this is a null action, expect it to be overloaded by specific algos
    def respond(self, time, lob, trade, verbose):
        return None

    # specify how trader mutates its parameter values
    # this is a null action, expect it to be overloaded by specific algos
    def mutate(self, time, lob, trade, verbose):
        return None

    def set_opinion(self, updated_opinion):

        validated_update = updated_opinion

        if updated_opinion >= self.upper_op_bound:
            # set to upper bound
            validated_update = self.upper_op_bound
        elif updated_opinion <= self.lower_op_bound:
            # set to lower bound
            validated_update = self.lower_op_bound

        self.opinion = validated_update

    def set_uncertainty(self, updated_uncertainty):

        validated_update = updated_uncertainty

        if updated_uncertainty >= self.upper_un_bound:
            # set to upper bound
            validated_update = self.upper_un_bound
        elif updated_uncertainty <= self.lower_un_bound:
            # set to lower bound
            validated_update = self.lower_un_bound

        self.uncertainty = validated_update

# the class that wraps PMFs for composition mechanisms
class prob_mass_fn:

    # here, generate a pmf based on a value (i.e., in calc_lut)

    def __init__(self, stratval, pmf1, pmf2, keyword, pmin, pmax, dirn):

        # creating a PMF when just given a float value (i.e., for strat and constants)
        if(keyword == 'null'):
            self.pmf = create_pmf(stratval, 100, 4, dirn, pmin, pmax)
            self.weight = 1
            self.stratval = stratval

        # creating a PMF combining 2 Prob mass fn objects in parallel
        if(keyword == 'parallel'):
            self.weight = 1
            self.pmf = PMF_joins.P_join(pmf1, pmf2)

        # creating a PMF combining 2 Prob mass fn objects in series
        if(keyword == 'series'):
            self.weight = 1
            self.pmf = PMF_joins.S_join(pmf1, pmf2)

        # creating a PMF combining 2 Prob mass fn objects in difference
        if(keyword == 'difference'):
            self.weight = 1
            self.pmf = PMF_joins.D_join(pmf1, pmf2)

    # generate a plot of the PMF
    def plot(self, color, fignum):

        df = pd.DataFrame(self.pmf['cdf_lut'])
        plt.plot(df["price"], df["prob"])
        plt.show()

    # used to update weightings of a PMF (used in s-join)
    def assign_weight(self, weight):
        self.weight = weight

    # returns the wrapped PMF
    def get_pmf(self):
        return self.pmf

    def get_weight(self):
        return self.weight

    def __call__(self):
        return self

# this function creates PMFs by just using a single float
def create_pmf(strat, t0, tang_m, dirn, pmin, pmax):
    # set parameter values and calculate CDF LUT
    # dirn is direction: -1 for buy, +1 for sell

    # the threshold function used to clip
    def threshold(theta0, x):
        t = max(-1*theta0, min(theta0, x))
        return t

    epsilon = 0.000001 #used to catch DIV0 errors
    verbose = False

    dxs = dirn * strat

    p_range = float(pmax - pmin)
    if p_range < 1:
        # special case: the SHVR-style strategy has shaved all the way to the limit price
        # the lower and upper bounds on the interval are adjacent prices;
        # so cdf is simply the lower price with probability 1
        cdf = []
        cdf.append({'price': pmin, 'prob': 0, 'cal_p': 0, 'cal_sum': 0, 'cum_prob': 1})

        return {'strat':strat, 'dirn':dirn, 'pmin':pmin, 'pmax':pmax, 'cdf_lut':cdf}

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

    # note, pmf also stores the cdf lut
    cdf = []
    cum_prob = 0
    # now go thru interval summing and normalizing to give the CDF
    # this is equation 12
    for p in range(pmin, pmax + 1):
        price = calp_interval[p-pmin]['price']
        cal_p = calp_interval[p-pmin]['cal_p']
        prob = cal_p / cal_sum                                               # this is the PMF
        cum_prob += prob
        cdf.append({'price': p, 'prob': prob, 'cal_p': cal_p, 'cal_sum': cal_sum, 'cum_prob': cum_prob})

    return {'strat':strat, 'dirn':dirn, 'pmin':pmin, 'pmax':pmax, 'cdf_lut':cdf}


# Trader subclass Giveaway
# even dumber than a ZI-U: just give the deal away
# (but never makes a loss)
class Trader_Giveaway(Trader):

    def getorder(self, time, countdown, lob):
        if len(self.orders) < 1:
            order = None
        else:
            quoteprice = self.orders[0].price
            order = Order(self.tid,
                          self.orders[0].otype,
                          quoteprice,
                          self.orders[0].qty,
                          time, lob['QID'])
            self.lastquote = order
        return order


# Trader subclass ZI-C
# After Gode & Sunder 1993
class Trader_ZIC(Trader):

    def getorder(self, time, countdown, lob):
        if len(self.orders) < 1:
            # no orders: return NULL
            order = None
        else:
            minprice = lob['bids']['worst']
            maxprice = lob['asks']['worst']
            qid = lob['QID']
            limit = self.orders[0].price
            otype = self.orders[0].otype
            if otype == 'Bid':
                quoteprice = random.randint(minprice, limit)
            else:
                quoteprice = random.randint(limit, maxprice)
                # NB should check it == 'Ask' and barf if not
            order = Order(self.tid, otype, quoteprice, self.orders[0].qty, time, qid)
            self.lastquote = order
        return order


# Trader subclass Shaver
# shaves a penny off the best price
# if there is no best price, creates "stub quote" at system max/min
class Trader_Shaver(Trader):

    def getorder(self, time, countdown, lob):
        if len(self.orders) < 1:
            order = None
        else:
            limitprice = self.orders[0].price
            otype = self.orders[0].otype
            if otype == 'Bid':
                if lob['bids']['n'] > 0:
                    quoteprice = lob['bids']['best'] + 1
                    if quoteprice > limitprice:
                        quoteprice = limitprice
                else:
                    quoteprice = lob['bids']['worst']
            else:
                if lob['asks']['n'] > 0:
                    quoteprice = lob['asks']['best'] - 1
                    if quoteprice < limitprice:
                        quoteprice = limitprice
                else:
                    quoteprice = lob['asks']['worst']
            order = Order(self.tid, otype, quoteprice, self.orders[0].qty, time, lob['QID'])
            self.lastquote = order
        return order


# Trader subclass Sniper
# Based on Shaver,
# "lurks" until time remaining < threshold% of the trading session
# then gets increasing aggressive, increasing "shave thickness" as time runs out
class Trader_Sniper(Trader):

    def getorder(self, time, countdown, lob):
        lurk_threshold = 0.2
        shavegrowthrate = 3
        shave = int(1.0 / (0.01 + countdown / (shavegrowthrate * lurk_threshold)))
        if (len(self.orders) < 1) or (countdown > lurk_threshold):
            order = None
        else:
            limitprice = self.orders[0].price
            otype = self.orders[0].otype

            if otype == 'Bid':
                if lob['bids']['n'] > 0:
                    quoteprice = lob['bids']['best'] + shave
                    if quoteprice > limitprice:
                        quoteprice = limitprice
                else:
                    quoteprice = lob['bids']['worst']
            else:
                if lob['asks']['n'] > 0:
                    quoteprice = lob['asks']['best'] - shave
                    if quoteprice < limitprice:
                        quoteprice = limitprice
                else:
                    quoteprice = lob['asks']['worst']
            order = Order(self.tid, otype, quoteprice, self.orders[0].qty, time, lob['QID'])
            self.lastquote = order
        return order


# Trader subclass PRZI
# added 23 March 2021
# Dave Cliff's Parameterized-Response Zero-Intelligence (PRZI) trader
# see https://arxiv.org/abs/2103.11341
class Trader_PRZI(Trader):

    def __init__(self, ttype, tid, balance, time, opinion, uncertainty, lower_op_bound, upper_op_bound, start_opinion):

        # PRZI strategy defined by parameter "strat"
        # here this is randomly assigned
        # strat * direction = -1 = > GVWY; =0 = > ZIC; =+1 = > SHVR

        Trader.__init__(self, ttype, tid, balance, time, opinion, uncertainty, lower_op_bound, upper_op_bound, start_opinion)
        self.theta0 = 100           # threshold-function limit value
        self.m = 4                  # tangent-function multiplier
        self.strat = 1.0 - 2 * random.random() # strategy parameter: must be in range [-1.0, +1.0]
        self.cdf_lut_bid = None     # look-up table for buyer cumulative distribution function
        self.cdf_lut_ask = None     # look-up table for buyer cumulative distribution function
        self.pmax = None            # this trader's estimate of the maximum price the market will bear
        self.pmax_c_i = math.sqrt(random.randint(1,10))  # multiplier coefficient when estimating p_max

    def getorder(self, time, countdown, lob):

        # shvr_price tells us what price a SHVR would quote in these circs
        def shvr_price(otype, limit, lob):

            if otype == 'Bid':
                if lob['bids']['n'] > 0:
                    shvr_p = lob['bids']['best'] + 1   # BSE tick size is always 1
                    if shvr_p > limit:
                        shvr_p = limit
                else:
                    shvr_p = lob['bids']['worst']
            else:
                if lob['asks']['n'] > 0:
                    shvr_p = lob['asks']['best'] - 1   # BSE tick size is always 1
                    if shvr_p < limit:
                        shvr_p = limit
                else:
                    shvr_p = lob['asks']['worst']

            return shvr_p

        # calculate cumulative distribution function (CDF) look-up table (LUT)
        def calc_cdf_lut(strat, t0, m, dirn, pmin, pmax):
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

            dxs = dirn * self.strat

            if verbose:
                print('calc_cdf_lut: dirn=%d dxs=%d pmin=%d pmax=%d\n' % (dirn, dxs, pmin, pmax))

            p_range = float(pmax - pmin)
            if p_range < 1:
                # special case: the SHVR-style strategy has shaved all the way to the limit price
                # the lower and upper bounds on the interval are adjacent prices;
                # so cdf is simply the lower price with probability 1

                cdf=[{'price':pmin, 'cum_prob': 1.0}]

                if verbose:
                    print('\n\ncdf:', cdf)

                return {'strat': strat, 'dirn': dirn, 'pmin': pmin, 'pmax': pmax, 'cdf_lut': cdf}

            c = threshold(t0, m * math.tan(math.pi * (strat + 0.5)))

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
            calp_sum = 0
            for p in range(pmin, pmax + 1):
                p_r = (p - pmin) / (p_range)  # p_r in [0.0, 1.0]
                if self.strat == 0.0:
                    # special case: this is just ZIC
                    cal_p = 1 / (p_range + 1)
                elif self.strat > 0:
                    cal_p = (math.exp(c * p_r) - 1.0) / e2cm1
                else:  # self.strat < 0
                    cal_p = 1.0 - ((math.exp(c * p_r) - 1.0) / e2cm1)
                if cal_p < 0:
                    cal_p = 0   # just in case
                calp_interval.append({'price':p, "cal_p":cal_p})
                calp_sum += cal_p

            if calp_sum <= 0:
                print('calp_interval:', calp_interval)
                print('pmin=%f, pmax=%f, calp_sum=%f' % (pmin, pmax, calp_sum))

            cdf = []
            cum_prob = 0
            # now go thru interval summing and normalizing to give the CDF
            for p in range(pmin, pmax + 1):
                price = calp_interval[p-pmin]['price']
                cal_p = calp_interval[p-pmin]['cal_p']
                prob = cal_p / calp_sum
                cum_prob += prob
                cdf.append({'price': p, 'cum_prob': cum_prob})

            if verbose:
                print('\n\ncdf:', cdf)

            return {'strat':strat, 'dirn':dirn, 'pmin':pmin, 'pmax':pmax, 'cdf_lut':cdf}

        verbose = False

        if verbose:
            print('PRZI getorder: strat=%f' % self.strat)

        if len(self.orders) < 1:
            # no orders: return NULL
            order = None
        else:
            # unpack the assignment-order
            limit = self.orders[0].price
            otype = self.orders[0].otype

            # get extreme limits on price interval
            # lowest price the market will bear
            minprice = int(lob['bids']['worst'])  # default assumption: worst bid price possible is 1 tick
            # trader's individual estimate highest price the market will bear
            if self.pmax is None:
                maxprice = int(limit * self.pmax_c_i + 0.5) # in the absence of any other info, guess
                self.pmax = maxprice
            elif self.pmax < lob['asks']['sess_hi']:        # some other trader has quoted higher than I expected
                maxprice = lob['asks']['sess_hi']           # so use that as my new estimate of highest
                self.pmax = maxprice
            else:
                maxprice = self.pmax

            # what price would a SHVR quote?
            p_shvr = shvr_price(otype, limit, lob)

            # it may be more efficient to detect the ZIC special case and generate a price directly
            # whether it is or not depends on how many entries need to be sampled in the LUT reverse-lookup
            # versus the compute time of the call to random.randint that would be used in direct ZIC
            # here, for simplicity, we're not treating ZIC as a special case...
            # ... so the full CDF LUT needs to be instantiated for ZIC (strat=0.0) just like any other strat value

            # use the cdf look-up table
            # cdf_lut is a list of little dictionaries
            # each dictionary has form: {'cum_prob':nnn, 'price':nnn}
            # generate u=U(0,1) uniform disrtibution
            # starting with the lowest nonzero cdf value at cdf_lut[0],
            # walk up the lut (i.e., examine higher cumulative probabilities),
            # until we're in the range of u; then return the relevant price


            # the LUTs are re-computed if any of the details have changed
            if otype == 'Bid':

                # direction * strat
                dxs = -1 * self.strat  # the minus one multiplier is the "buy" direction

                p_max = int(limit)
                if dxs <= 0:
                    p_min = minprice        # this is delta_p for BSE, i.e. ticksize =1
                else:
                    # shade the lower bound on the interval
                    # away from minprice and toward shvr_price
                    p_min = int(0.5 + (dxs * p_shvr) + ((1.0-dxs) * minprice))

                if (self.cdf_lut_bid is None) or \
                        (self.cdf_lut_bid['strat'] != self.strat) or\
                        (self.cdf_lut_bid['pmin'] != p_min) or \
                        (self.cdf_lut_bid['pmax'] != p_max):
                    # need to compute a new LUT
                    if verbose:
                        print('New bid LUT')
                    self.cdf_lut_bid = calc_cdf_lut(self.strat, self.theta0,
                                                    self.m, -1, p_min, p_max)

                lut = self.cdf_lut_bid

            else:   # otype == 'Ask'

                dxs = self.strat

                p_min = int(limit)
                if dxs <= 0:
                    p_max = maxprice
                else:
                    # shade the upper bound on the interval
                    # away from maxprice and toward shvr_price
                    p_max = int(0.5 + (dxs * p_shvr) + ((1.0-dxs) * maxprice))

                if (self.cdf_lut_ask is None) or \
                        (self.cdf_lut_ask['strat'] != self.strat) or \
                        (self.cdf_lut_ask['pmin'] != p_min) or \
                        (self.cdf_lut_ask['pmax'] != p_max):
                    # need to compute a new LUT
                    if verbose:
                        print('New ask LUT')
                    self.cdf_lut_ask = calc_cdf_lut(self.strat, self.theta0,
                                                    self.m, +1, p_min, p_max)

                lut = self.cdf_lut_ask

            if verbose:
                print('PRZI LUT =', lut)

            # do inverse lookup on the LUT to find the price
            u = random.random()
            for entry in lut['cdf_lut']:
                if u < entry['cum_prob']:
                    quoteprice = entry['price']
                    break

            order = Order(self.tid, otype,
                          quoteprice, self.orders[0].qty, time, lob['QID'])

            self.lastquote = order

        return order


# Trader subclass ZIP
# After Cliff 1997
class Trader_ZIP(Trader):

    # ZIP init key param-values are those used in Cliff's 1997 original HP Labs tech report
    # NB this implementation keeps separate margin values for buying & selling,
    #    so a single trader can both buy AND sell
    #    -- in the original, traders were either buyers OR sellers

    def __init__(self, ttype, tid, balance, time, opinion, uncertainty, lower_op_bound, upper_op_bound, start_opinion):
        Trader.__init__(self, ttype, tid, balance, time, opinion, uncertainty, lower_op_bound, upper_op_bound, start_opinion)
        self.willing = 1
        self.able = 1
        self.job = None  # this gets switched to 'Bid' or 'Ask' depending on order-type
        self.active = False  # gets switched to True while actively working an order
        self.prev_change = 0  # this was called last_d in Cliff'97
        self.beta = 0.1 + 0.4 * random.random()
        self.momntm = 0.1 * random.random()
        self.ca = 0.05  # self.ca & .cr were hard-coded in '97 but parameterised later
        self.cr = 0.05
        self.margin = None  # this was called profit in Cliff'97
        self.margin_buy = -1.0 * (0.05 + 0.3 * random.random())
        self.margin_sell = 0.05 + 0.3 * random.random()
        self.price = None
        self.limit = None
        # memory of best price & quantity of best bid and ask, on LOB on previous update
        self.prev_best_bid_p = None
        self.prev_best_bid_q = None
        self.prev_best_ask_p = None
        self.prev_best_ask_q = None

        self.opinion = opinion        # opinion between [0,1]
        self.uncertainty = uncertainty # uncertainty between [0, 2]

        self.lower_op_bound = lower_op_bound
        self.upper_op_bound = upper_op_bound
        self.lower_un_bound = 0
        self.upper_un_bound = 2

        self.start_opinion = start_opinion
        self.n_iter = 0

    def getorder(self, time, countdown, lob):
        if len(self.orders) < 1:
            self.active = False
            order = None
        else:
            self.active = True
            self.limit = self.orders[0].price
            self.job = self.orders[0].otype
            if self.job == 'Bid':
                # currently a buyer (working a bid order)
                self.margin = self.margin_buy
            else:
                # currently a seller (working a sell order)
                self.margin = self.margin_sell
            quoteprice = int(self.limit * (1 + self.margin))
            self.price = quoteprice

            order = Order(self.tid, self.job, quoteprice, self.orders[0].qty, time, lob['QID'])
            self.lastquote = order
        return order

    # update margin on basis of what happened in market
    def respond(self, time, lob, trade, verbose):
        # ZIP trader responds to market events, altering its margin
        # does this whether it currently has an order to work or not

        def target_up(price):
            # generate a higher target price by randomly perturbing given price
            ptrb_abs = self.ca * random.random()  # absolute shift
            ptrb_rel = price * (1.0 + (self.cr * random.random()))  # relative shift
            target = int(round(ptrb_rel + ptrb_abs, 0))
            # #                        print('TargetUp: %d %d\n' % (price,target))
            return target

        def target_down(price):
            # generate a lower target price by randomly perturbing given price
            ptrb_abs = self.ca * random.random()  # absolute shift
            ptrb_rel = price * (1.0 - (self.cr * random.random()))  # relative shift
            target = int(round(ptrb_rel - ptrb_abs, 0))
            # #                        print('TargetDn: %d %d\n' % (price,target))
            return target

        def willing_to_trade(price):
            # am I willing to trade at this price?
            willing = False
            if self.job == 'Bid' and self.active and self.price >= price:
                willing = True
            if self.job == 'Ask' and self.active and self.price <= price:
                willing = True
            return willing

        def profit_alter(price):
            oldprice = self.price
            diff = price - oldprice
            change = ((1.0 - self.momntm) * (self.beta * diff)) + (self.momntm * self.prev_change)
            self.prev_change = change
            newmargin = ((self.price + change) / self.limit) - 1.0

            if self.job == 'Bid':
                if newmargin < 0.0:
                    self.margin_buy = newmargin
                    self.margin = newmargin
            else:
                if newmargin > 0.0:
                    self.margin_sell = newmargin
                    self.margin = newmargin

            # set the price from limit and profit-margin
            self.price = int(round(self.limit * (1.0 + self.margin), 0))

        # #                        print('old=%d diff=%d change=%d price = %d\n' % (oldprice, diff, change, self.price))

        # what, if anything, has happened on the bid LOB?
        bid_improved = False
        bid_hit = False
        lob_best_bid_p = lob['bids']['best']
        lob_best_bid_q = None
        if lob_best_bid_p is not None:
            # non-empty bid LOB
            lob_best_bid_q = lob['bids']['lob'][-1][1]
            if (self.prev_best_bid_p is not None) and (self.prev_best_bid_p < lob_best_bid_p):
                # best bid has improved
                # NB doesn't check if the improvement was by self
                bid_improved = True
            elif trade is not None and ((self.prev_best_bid_p > lob_best_bid_p) or (
                    (self.prev_best_bid_p == lob_best_bid_p) and (self.prev_best_bid_q > lob_best_bid_q))):
                # previous best bid was hit
                bid_hit = True
        elif self.prev_best_bid_p is not None:
            # the bid LOB has been emptied: was it cancelled or hit?
            last_tape_item = lob['tape'][-1]
            if last_tape_item['type'] == 'Cancel':
                bid_hit = False
            else:
                bid_hit = True

        # what, if anything, has happened on the ask LOB?
        ask_improved = False
        ask_lifted = False
        lob_best_ask_p = lob['asks']['best']
        lob_best_ask_q = None
        if lob_best_ask_p is not None:
            # non-empty ask LOB
            lob_best_ask_q = lob['asks']['lob'][0][1]
            if (self.prev_best_ask_p is not None) and (self.prev_best_ask_p > lob_best_ask_p):
                # best ask has improved -- NB doesn't check if the improvement was by self
                ask_improved = True
            elif trade is not None and ((self.prev_best_ask_p < lob_best_ask_p) or (
                    (self.prev_best_ask_p == lob_best_ask_p) and (self.prev_best_ask_q > lob_best_ask_q))):
                # trade happened and best ask price has got worse, or stayed same but quantity reduced
                # -- assume previous best ask was lifted
                ask_lifted = True
        elif self.prev_best_ask_p is not None:
            # the ask LOB is empty now but was not previously: canceled or lifted?
            last_tape_item = lob['tape'][-1]
            if last_tape_item['type'] == 'Cancel':
                ask_lifted = False
            else:
                ask_lifted = True

        if verbose and (bid_improved or bid_hit or ask_improved or ask_lifted):
            print('B_improved', bid_improved, 'B_hit', bid_hit, 'A_improved', ask_improved, 'A_lifted', ask_lifted)

        deal = bid_hit or ask_lifted

        if self.job == 'Ask':
            # seller
            if deal:
                tradeprice = trade['price']
                if self.price <= tradeprice:
                    # could sell for more? raise margin
                    target_price = target_up(tradeprice)
                    profit_alter(target_price)
                elif ask_lifted and self.active and not willing_to_trade(tradeprice):
                    # wouldnt have got this deal, still working order, so reduce margin
                    target_price = target_down(tradeprice)
                    profit_alter(target_price)
            else:
                # no deal: aim for a target price higher than best bid
                if ask_improved and self.price > lob_best_ask_p:
                    if lob_best_bid_p is not None:
                        target_price = target_up(lob_best_bid_p)
                    else:
                        target_price = lob['asks']['worst']  # stub quote
                    profit_alter(target_price)

        if self.job == 'Bid':
            # buyer
            if deal:
                tradeprice = trade['price']
                if self.price >= tradeprice:
                    # could buy for less? raise margin (i.e. cut the price)
                    target_price = target_down(tradeprice)
                    profit_alter(target_price)
                elif bid_hit and self.active and not willing_to_trade(tradeprice):
                    # wouldnt have got this deal, still working order, so reduce margin
                    target_price = target_up(tradeprice)
                    profit_alter(target_price)
            else:
                # no deal: aim for target price lower than best ask
                if bid_improved and self.price < lob_best_bid_p:
                    if lob_best_ask_p is not None:
                        target_price = target_down(lob_best_ask_p)
                    else:
                        target_price = lob['bids']['worst']  # stub quote
                    profit_alter(target_price)

        # remember the best LOB data ready for next response
        self.prev_best_bid_p = lob_best_bid_p
        self.prev_best_bid_q = lob_best_bid_q
        self.prev_best_ask_p = lob_best_ask_p
        self.prev_best_ask_q = lob_best_ask_q

##################--Opinionated Traders below here--#############

class Trader_opinionated_ZIC(Trader):

    def getorder(self, time, countdown, lob):

        if len(self.orders) < 1:
            # no orders: return NULL
            order = None
        else:
            minprice = lob['bids']['worst']
            maxprice = lob['asks']['worst']
            qid = lob['QID']
            limit = self.orders[0].price
            otype = self.orders[0].otype

            # set opinonated limit price using own opinion
            # opinionated_limit = int(limit + self.opinion * 100)

            # buyer
            if otype == 'Bid':

                opinionated_limit = int((limit * ( 1 + self.opinion) + minprice * (1 - self.opinion))/2)
                # opinionated_limit = limit
                # quotelimit = max(min(int(limit + self.opinion * 100), limit), minprice)

                # print("BID QUOTE: min: %d max: %d | omax-2: %d omax-1: %d" % (minprice, limit, opinionated_limit, quotelimit))
                # print("BID QUOTE: limit %d, minprice %d, opinionated_limit %d, quote limit %d " % (self.orders[0].price, minprice, opinionated_limit, limit))
                quoteprice = random.randint(minprice, opinionated_limit)
                # print("BID: %d %d %d" % (minprice, limit, quoteprice))

            # seller
            else:

                opinionated_limit = int((limit * ( 1 - self.opinion) + maxprice * (1 + self.opinion))/2)

                # quotelimit = min(max(int(limit + self.opinion * 100), limit), maxprice)

                # print("ASK QUOTE: min: %d max: %d | omin-2: %d omin-1: %d" % (limit, maxprice, opinionated_limit, quotelimit))
                # print("ASK QUOTE: limit %d, maxprice %d, opinionated_limit %d, quote limit %d " % (self.orders[0].price, maxprice, opinionated_limit, limit))
                quoteprice = random.randint(opinionated_limit, maxprice)
                # print("ASK: %d %d %d" % (limit, maxprice, quoteprice))
                # NB should check it == 'Ask' and barf if not
            order = Order(self.tid, otype, quoteprice, self.orders[0].qty, time, qid)
            self.lastquote = order

        return order




######################################### OPRZI

######################################### OPRZI

class Trader_OPRZI(Trader):

    def __init__(self, ttype, tid, balance, time, m, opinion, uncertainty, lower_op_bound, upper_op_bound, start_opinion, truth_prevalence):
        # PRZI strategy defined by parameter "strat"
        # here this is randomly assigned
        # strat * direction = -1 = > GVWY; =0 = > ZIC; =+1 = > SHVR
        # note, opinion, uncertainty etc all calculated before initialisation and passed into init function

        Trader.__init__(self, ttype, tid, balance, time, opinion, uncertainty, lower_op_bound, upper_op_bound, start_opinion)
        self.theta0 = 100           # threshold-function limit value
        self.tang_m = 4                  # tangent-function multiplier
        self.strat = 1.0 - 2 * random.random() # strategy parameter: must be in range [-1.0, +1.0]
        self.cdf_lut_bid = None     # look-up table for buyer cumulative distribution function
        self.cdf_lut_ask = None     # look-up table for buyer cumulative distribution function
        self.pmax = None            # this trader's estimate of the maximum price the market will bear
        self.pmax_c_i = math.sqrt(random.randint(1,10))  # multiplier coefficient when estimating p_max

        # Kenny opinion stuff
        self.opinion = opinion        # opinion between [0,1]
        self.uncertainty = uncertainty # uncertainty between [0, 2]

        self.lower_op_bound = lower_op_bound
        self.upper_op_bound = upper_op_bound
        self.lower_un_bound = 0
        self.upper_un_bound = 2

        self.start_opinion = start_opinion
        self.n_iter = 0

        #UNCOMMENT

        # variables for MLOFI
        self.last_lob = None;
        self.es_list = [];
        self.ds_list = [];

        #variable for ratio
        self.bids_volume_list = []
        self.asks_volume_list = []

        # m
        self.m = m;

        # value for ω, i.e., how much weight on global opinion - their "weight" on the market data
        self.wmkt = truth_prevalence

        # variables for shocks
        self.shockActivated = False
        self.mlofiShockValue = None
        self.locOpShockValue = None
        # stores tendency of this intelligent extremist
        # if this value is not None, they are an intelligent extremist
        self.intExtremistValue = None
        self.time_activated = None


    ##################################
    ## Functions needed for shocks ###
    ##################################

    def execMlofiShock(self, value):
        if(type(value) != int and type(value) != float):
            sys.exit('Invalid MLOFI offset shock value - must be int/float')
        self.mlofiShockValue = value
        self.shockActivated = True

    def execLocOpShock(self, value):
        if(type(value) != int and type(value) != float):
            sys.exit('Invalid Local Opinion shock value - must be int/float btw -1 and 1')
        if(value > 1 or value < -1):
            sys.exit('Invalid Local Opinion shock value - must be int/float btw -1 and 1')
        self.locOpShockValue = value
        self.shockActivated = True

    def activateIntExtremists(self, value, time_activated):
        if(value > 1 or value < -1):
            sys.exit('Cannot have a tendency greater than op bounds')
        self.intExtremistValue = value
        self.shockActivated = True
        self.time_activated = time_activated

    ##################################
    ## Functions needed to respond ###
    ##################################

    def respond(self, time, lob, trade, verbose):
        # when a market event occurs, respond is called on all traders
        # O-PRZI response is to update its lists that are used to calc MLOFI

        if (self.last_lob == None):
            self.last_lob = lob
        else:
            self.calc_es(lob, self.m, verbose)
            self.calc_ds(lob, self.m, verbose)
            self.calc_bids_volume(lob, self.m, verbose)
            self.calc_asks_volume(lob, self.m, verbose)
            self.last_lob = lob

    def calc_bids_volume(self, lob, m, verbose):
        new_b = {}

        for i in range(1, m + 1):
            new_b['level' + str(i)] = self.cal_bids_n(lob, i)

        self.bids_volume_list.append(new_b)

    def cal_bids_n(self, lob, n):

        if (len(lob['bids']['lob']) < n):
            r_n = 0
        else:
            r_n = lob['bids']['lob'][n - 1][1]

        return r_n

    def calc_asks_volume(self, lob, m, verbose):

        new_a = {}

        for i in range(1, m + 1):
            new_a['level' + str(i)] = self.cal_asks_n(lob, i);

        self.asks_volume_list.append(new_a)

    def cal_asks_n(self, lob, n):

        if (len(lob['asks']['lob']) < n):
            q_n = 0
        else:
            q_n = lob['asks']['lob'][n - 1][1]
        return q_n

    def calc_level_n_e(self, current_lob, n):
        b_n = 0
        r_n = 0
        a_n = 0
        q_n = 0

        b_n_1 = 0
        r_n_1 = 0
        a_n_1 = 0
        q_n_1 = 0

        if (len(current_lob['bids']['lob']) >= n):
            b_n = current_lob['bids']['lob'][n - 1][0]
            r_n = current_lob['bids']['lob'][n - 1][1]

        if (len(self.last_lob['bids']['lob']) >= n):
            b_n_1 = self.last_lob['bids']['lob'][n - 1][0]
            r_n_1 = self.last_lob['bids']['lob'][n - 1][1]

        if (len(current_lob['asks']['lob']) >= n):
            a_n = current_lob['asks']['lob'][n - 1][0]
            q_n = current_lob['asks']['lob'][n - 1][1]

        if (len(self.last_lob['asks']['lob']) >= n):
            a_n_1 = self.last_lob['asks']['lob'][n - 1][0]
            q_n_1 = self.last_lob['asks']['lob'][n - 1][1]

        delta_w = 0

        if (b_n > b_n_1):
            delta_w = r_n
        elif (b_n == b_n_1):
            delta_w = r_n - r_n_1
        else:
            delta_w = -r_n_1

        delta_v = 0

        if (a_n > a_n_1):
            delta_v = -q_n_1
        elif (a_n == a_n_1):
            delta_v = q_n - q_n_1
        else:
            delta_v = q_n

        return delta_w - delta_v

    def calc_es(self, lob, m, verbose):
        new_e = {}
        for i in range(1, m + 1):
            new_e['level' + str(i)] = self.calc_level_n_e(lob, i)

        self.es_list.append(new_e)

    def calc_ds(self, lob, m, verbose):
        new_d = {}

        for i in range(1, m + 1):
            new_d['level' + str(i)] = self.cal_depth_n(lob, i)

        self.ds_list.append(new_d)

    def cal_depth_n(self, lob, n):

        if (len(lob['bids']['lob']) < n):
            r_n = 0
        else:
            r_n = lob['bids']['lob'][n - 1][1]

        if (len(lob['asks']['lob']) < n):
            q_n = 0
        else:
            q_n = lob['asks']['lob'][n - 1][1]
        return (r_n + q_n) / 2

    # opinion dynamics update of opinions
    def set_opinion(self, updated_opinion):

        validated_update = updated_opinion

        if updated_opinion >= self.upper_op_bound:
            # set to upper bound
            validated_update = self.upper_op_bound
        elif updated_opinion <= self.lower_op_bound:
            # set to lower bound
            validated_update = self.lower_op_bound

        self.opinion = validated_update

    # ensures normalised offset is between -1 and 1
    def sigmoid(self, num):

        #was 5.5
        return 2 * (1/(1+math.exp((-10.5)*num)) - 0.5)

    # update of local opinion using global opinion
    def adjust_strat(self):

        # if there is an imbalance, use the weighted average

        if(self.theta != 0):
            strat_val = (self.theta * self.wmkt) + ((1.0-self.wmkt) * self.opinion)

        # if there is no imbalance, trader solely off of opinion

        else:
            strat_val = self.opinion

        return strat_val

    def get_offset(self, lob, countdown, m):
        # a list to store MLOFI from level 1 to level m
        mlofi_list = [0 for i in range(m)]

        # a list to store the cumulative depth from level 1 to level m
        cd_list = [0 for i in range(m)]

        # a list to store the average depth from level 1 to level m
        ad_list = []

        n = 1

        # calculate MLOFI in each level
        while len(self.es_list) >= n:
            for i in range(m):
                mlofi_list[i] += self.es_list[-n]['level' + str(i+1)]
            n += 1
            # consider at most last 10 events
            if n >= 11:
                break

        n = 1
        # aggregate depths in each level
        while len(self.ds_list) >= n:
            for i in range(m):
                cd_list[i] += self.ds_list[-n]['level' + str(i+1)]
            n += 1
            # consider at most last 10 events
            if n >= 11:
                break

        # calculate average depth in each level
        for i in range(m):
            temp = None
            if n == 1:
                temp = cd_list[i]+1
            else:
                temp = cd_list[i]/(n-1)+1
            ad_list.append(temp)

        c = 5
        decay = 0.8
        offset = 0

        # calculate offset
        for i in range(m):
            offset += int(mlofi_list[i]*c*pow(decay,i)/ ad_list[i])

        return offset


    def set_uncertainty(self, updated_uncertainty):

        validated_update = updated_uncertainty

        if updated_uncertainty >= self.upper_un_bound:
            # set to upper bound
            validated_update = self.upper_un_bound
        elif updated_uncertainty <= self.lower_un_bound:
            # set to lower bound
            validated_update = self.lower_un_bound

        self.uncertainty = validated_update

    def calc_es(self, lob, m, verbose):
        new_e = {}
        for i in range(1, m + 1):
            new_e['level' + str(i)] = self.calc_level_n_e(lob, i)

        self.es_list.append(new_e)

    def getorder(self, time, countdown, lob):

        # shvr_price tells us what price a SHVR would quote in these circs
        def shvr_price(otype, limit, lob):

            if otype == 'Bid':
                if lob['bids']['n'] > 0:
                    shvr_p = lob['bids']['best'] + 1   # BSE tick size is always 1
                    if shvr_p > limit:
                        shvr_p = limit
                else:
                    shvr_p = lob['bids']['worst']
            else:
                if lob['asks']['n'] > 0:
                    shvr_p = lob['asks']['best'] - 1   # BSE tick size is always 1
                    if shvr_p < limit:
                        shvr_p = limit
                else:
                    shvr_p = lob['asks']['worst']

            return shvr_p


        # calculate cumulative distribution function (CDF) look-up table (LUT)
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

            dxs = dirn * self.strat

            if verbose:
                print('calc_cdf_lut: dirn=%d dxs=%d pmin=%d pmax=%d\n' % (dirn, dxs, pmin, pmax))

            p_range = float(pmax - pmin)
            if p_range < 1:
                # special case: the SHVR-style strategy has shaved all the way to the limit price
                # the lower and upper bounds on the interval are adjacent prices;
                # so cdf is simply the lower price with probability 1

                cdf=[{'price':pmin, 'cum_prob': 1.0}]

                if verbose:
                    print('\n\ncdf:', cdf)

                return {'strat': strat, 'dirn': dirn, 'pmin': pmin, 'pmax': pmax, 'cdf_lut': cdf}

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
            calp_sum = 0
            for p in range(pmin, pmax + 1):
                p_r = (p - pmin) / (p_range)  # p_r in [0.0, 1.0]
                if self.strat == 0.0:
                    # special case: this is just ZIC
                    cal_p = 1 / (p_range + 1)
                elif self.strat > 0:
                    cal_p = (math.exp(c * p_r) - 1.0) / e2cm1
                else:  # self.strat < 0
                    cal_p = 1.0 - ((math.exp(c * p_r) - 1.0) / e2cm1)
                if cal_p < 0:
                    cal_p = 0   # just in case
                calp_interval.append({'price':p, "cal_p":cal_p})
                calp_sum += cal_p

            if calp_sum <= 0:
                print('calp_interval:', calp_interval)
                print('pmin=%f, pmax=%f, calp_sum=%f' % (pmin, pmax, calp_sum))

            cdf = []
            cum_prob = 0
            # now go thru interval summing and normalizing to give the CDF
            for p in range(pmin, pmax + 1):
                price = calp_interval[p-pmin]['price']
                cal_p = calp_interval[p-pmin]['cal_p']
                prob = cal_p / calp_sum
                cum_prob += prob
                cdf.append({'price': p, 'cum_prob': cum_prob})

            if verbose:
                print('\n\ncdf:', cdf)

            return {'strat':strat, 'dirn':dirn, 'pmin':pmin, 'pmax':pmax, 'cdf_lut':cdf}

        verbose = False

        if verbose:
            print('PRZI getorder: strat=%f' % self.strat)

        if len(self.orders) < 1:
            # no orders: return NULL
            order = None
        else:
            # unpack the assignment-order
            limit = self.orders[0].price
            otype = self.orders[0].otype

            microp = lob['microprice']
            midp = lob['midprice']

            # get extreme limits on price interval
            # lowest price the market will bear
            minprice = int(lob['bids']['worst'])  # default assumption: worst bid price possible is 1 tick
            # trader's individual estimate highest price the market will bear
            if self.pmax is None:
                maxprice = int(limit * self.pmax_c_i + 0.5) # in the absence of any other info, guess
                self.pmax = maxprice
            elif lob['asks']['sess_hi'] is not None:
                if self.pmax < lob['asks']['sess_hi']:        # some other trader has quoted higher than I expected
                    maxprice = lob['asks']['sess_hi']           # so use that as my new estimate of highest
                    self.pmax = maxprice
                else:
                    maxprice = self.pmax
            else:
                maxprice = self.pmax

            # what price would a SHVR quote?
            p_shvr = shvr_price(otype, limit, lob)

            # before we do anything, need to update opinion based on market
            # here is where we calculate MLOFI-offset

            offset = self.get_offset(lob, countdown, self.m)

            # sometimes microprice doesn't exist
            # in that case, use pmax to normalise, else use microprice
            try:
                offset /= lob['microprice']
            except:
                offset /= self.pmax

            # put through sigmoid function
            self.theta = self.sigmoid(offset)

            # calculate new strat value = new weighted opinion
            self.strat = self.adjust_strat()


            # if this trader is an intelligent extremist, they will just set
            # their composite opinion and strategy equal to the opinion tendency value

            if(self.intExtremistValue is not None):
                if(self.intExtremistValue > 0):
                    k = 0.002
                    self.opinion = min(1,(k*(time-self.time_activated)))
                    self.strat = self.opinion
                if(self.intExtremistValue < 0):
                    k = -0.002
                    self.opinion = max(-1,(k*(time-self.time_activated)))
                    self.strat = self.opinion

            # to reduce the impact of a largely decreasing pmax as s approaches 1
            # and a largely increasing pmin as s approaches -1, evaluate to 1
            if(self.strat >= 0.99):
                self.strat = 1
            if(self.strat <= -0.99):
                self.strat = -1

            # opinion is set equal to its strat
            self.opinion = self.strat

            # the LUTs are re-computed if any of the details have changed
            if otype == 'Bid':

                # direction * strat
                dxs = -1 * self.strat  # the minus one multiplier is the "buy" direction

                p_max = int(limit)

                if dxs <= 0:
                    p_min = minprice        # this is delta_p for BSE, i.e. ticksize =1
                else:
                    # shade the lower bound on the interval
                    # away from minprice and toward shvr_price
                    p_min = int(0.5 + (dxs * p_shvr) + ((1.0-dxs) * minprice))

                if(dxs == -1):
                    p_min = p_max           # if pure-gvwy strategy, set p_min = p_max

                if(dxs == 1):
                    p_max = p_min       # if pure-shvr strategy, set p_max = p_max = shvr price

                if (self.cdf_lut_bid is None) or \
                        (self.cdf_lut_bid['strat'] != self.strat) or\
                        (self.cdf_lut_bid['pmin'] != p_min) or \
                        (self.cdf_lut_bid['pmax'] != p_max):
                    # need to compute a new LUT
                    if verbose:
                        print('New bid LUT')
                    self.cdf_lut_bid = calc_cdf_lut(self.strat, self.theta0,
                                                    self.tang_m, -1, p_min, p_max)

                lut = self.cdf_lut_bid

            else:   # otype == 'Ask'

                dxs = self.strat

                p_min = int(limit)

                if dxs <= 0:
                    p_max = maxprice
                else:
                    # shade the upper bound on the interval
                    # away from maxprice and toward shvr_price
                    p_max = int(0.5 + (dxs * p_shvr) + ((1.0-dxs) * maxprice))

                if(dxs == 1):
                    p_min = p_max           # if pure-shaver strategy, set p_min = p_max

                if(dxs == -1):
                    p_max = p_min           # if pure-gvwy strategy, set p_max = p_min = limit price

                if (self.cdf_lut_ask is None) or \
                        (self.cdf_lut_ask['strat'] != self.strat) or \
                        (self.cdf_lut_ask['pmin'] != p_min) or \
                        (self.cdf_lut_ask['pmax'] != p_max):
                    # need to compute a new LUT
                    if verbose:
                        print('New ask LUT')
                    self.cdf_lut_ask = calc_cdf_lut(self.strat, self.theta0,
                                                    self.tang_m, +1, p_min, p_max)

                lut = self.cdf_lut_ask

            if verbose:
                print('PRZI LUT =', lut)

            # do inverse lookup on the LUT to find the price
            u = random.random()
            for entry in lut['cdf_lut']:
                if u < entry['cum_prob']:
                    quoteprice = entry['price']
                    break

            order = Order(self.tid, otype,
                          quoteprice, self.orders[0].qty, time, lob['QID'])

            self.lastquote = order

            return order

####################################################################################
# BSE_GP trader defined here

# note, this trader is slow to calculate its PMFs, hence a lot of these traders
# result in slow simulations - however, execution performance could be improved
# by calculating a DB full of OPRZI_GP LUTs for different key values of (pmin, pmax, strat)
# however, this was out of the scope of this project

class Trader_OPRZI_GP(Trader):

    def __init__(self, ttype, tid, balance, time, m, opinion, uncertainty, lower_op_bound, upper_op_bound, start_opinion,  truth_prevalence):
        # PRZI strategy defined by parameter "strat"
        # here this is randomly assigned
        # strat * direction = -1 = > GVWY; =0 = > ZIC; =+1 = > SHVR
        # note, opinion, uncertainty etc all calculated before initialisation and passed into init function

        Trader.__init__(self, ttype, tid, balance, time, opinion, uncertainty, lower_op_bound, upper_op_bound, start_opinion)
        self.theta0 = 100           # threshold-function limit value
        self.tang_m = 4                  # tangent-function multiplier
        self.strat = 1.0 - 2 * random.random() # strategy parameter: must be in range [-1.0, +1.0]
        self.cdf_lut_bid = None     # look-up table for buyer cumulative distribution function
        self.cdf_lut_ask = None     # look-up table for buyer cumulative distribution function
        self.pmax = None            # this trader's estimate of the maximum price the market will bear
        self.pmax_c_i = math.sqrt(random.randint(1,10))  # multiplier coefficient when estimating p_max

        # Kenny opinion stuff
        self.opinion = opinion        # opinion between [0,1]
        self.uncertainty = uncertainty # uncertainty between [0, 2]

        self.lower_op_bound = lower_op_bound
        self.upper_op_bound = upper_op_bound
        self.lower_un_bound = 0
        self.upper_un_bound = 2

        self.start_opinion = start_opinion
        self.n_iter = 0

        # variable for MLOFI
        self.last_lob = None;
        self.es_list = [];
        self.ds_list = [];

        #variable for ratio
        self.bids_volume_list = []
        self.asks_volume_list = []

        # m
        self.m = m;

        # value for ω, i.e., how much weight on global opinion - their "weight" on the market data
        self.wmkt = truth_prevalence

        #vars for pmf composition
        self.updated_pmin = None
        self.updated_pmax = None
        self.current_trader_dir = None

        # variables for shocks
        self.shockActivated = False
        self.mlofiShockValue = None
        self.locOpShockValue = None
        # stores tendency of this intelligent extremist
        # if this value is not None, they are an intelligent extremist
        self.intExtremistValue = None
        self.time_activated = None

    ##################################
    ## Functions needed to respond ###
    ##################################

    ##################################
    ## Functions needed for shocks ###
    ##################################

    def execMlofiShock(self, value):
        if(type(value) != int and type(value) != float):
            sys.exit('Invalid MLOFI offset shock value - must be int/float')
        self.mlofiShockValue = value
        self.shockActivated = True

    def execLocOpShock(self, value):
        if(type(value) != int and type(value) != float):
            sys.exit('Invalid Local Opinion shock value - must be int/float btw -1 and 1')
        if(value > 1 or value < -1):
            sys.exit('Invalid Local Opinion shock value - must be int/float btw -1 and 1')
        self.locOpShockValue = value
        self.shockActivated = True

    def activateIntExtremists(self, value, time_activated):
        if(value > 1 or value < -1):
            sys.exit('Cannot have a tendency greater than op bounds')
        self.intExtremistValue = value
        self.shockActivated = True
        self.time_activated = time_activated

    def respond(self, time, lob, trade, verbose):

        # when a market event occurs, respond is called on all traders
        # O-PRZI response is to update its lists that are used to calc MLOFI

        if (self.last_lob == None):
            self.last_lob = lob
        else:
            self.calc_es(lob, self.m, verbose)
            self.calc_ds(lob, self.m, verbose)
            self.calc_bids_volume(lob, self.m, verbose)
            self.calc_asks_volume(lob, self.m, verbose)
            self.last_lob = lob

    def calc_bids_volume(self, lob, m, verbose):
        new_b = {}

        for i in range(1, m + 1):
            new_b['level' + str(i)] = self.cal_bids_n(lob, i)

        self.bids_volume_list.append(new_b)

    def cal_bids_n(self, lob, n):

        if (len(lob['bids']['lob']) < n):
            r_n = 0
        else:
            r_n = lob['bids']['lob'][n - 1][1]

        return r_n

    def calc_asks_volume(self, lob, m, verbose):

        new_a = {}

        for i in range(1, m + 1):
            new_a['level' + str(i)] = self.cal_asks_n(lob, i);

        self.asks_volume_list.append(new_a)

    def cal_asks_n(self, lob, n):

        if (len(lob['asks']['lob']) < n):
            q_n = 0
        else:
            q_n = lob['asks']['lob'][n - 1][1]
        return q_n

    def calc_level_n_e(self, current_lob, n):
        b_n = 0
        r_n = 0
        a_n = 0
        q_n = 0

        b_n_1 = 0
        r_n_1 = 0
        a_n_1 = 0
        q_n_1 = 0

        if (len(current_lob['bids']['lob']) >= n):
            b_n = current_lob['bids']['lob'][n - 1][0]
            r_n = current_lob['bids']['lob'][n - 1][1]

        if (len(self.last_lob['bids']['lob']) >= n):
            b_n_1 = self.last_lob['bids']['lob'][n - 1][0]
            r_n_1 = self.last_lob['bids']['lob'][n - 1][1]

        if (len(current_lob['asks']['lob']) >= n):
            a_n = current_lob['asks']['lob'][n - 1][0]
            q_n = current_lob['asks']['lob'][n - 1][1]

        if (len(self.last_lob['asks']['lob']) >= n):
            a_n_1 = self.last_lob['asks']['lob'][n - 1][0]
            q_n_1 = self.last_lob['asks']['lob'][n - 1][1]

        delta_w = 0

        if (b_n > b_n_1):
            delta_w = r_n
        elif (b_n == b_n_1):
            delta_w = r_n - r_n_1
        else:
            delta_w = -r_n_1

        delta_v = 0

        if (a_n > a_n_1):
            delta_v = -q_n_1
        elif (a_n == a_n_1):
            delta_v = q_n - q_n_1
        else:
            delta_v = q_n

        return delta_w - delta_v

    def calc_es(self, lob, m, verbose):
        new_e = {}
        for i in range(1, m + 1):
            new_e['level' + str(i)] = self.calc_level_n_e(lob, i)

        self.es_list.append(new_e)

    def calc_ds(self, lob, m, verbose):
        new_d = {}

        for i in range(1, m + 1):
            new_d['level' + str(i)] = self.cal_depth_n(lob, i)

        self.ds_list.append(new_d)

    def cal_depth_n(self, lob, n):

        if (len(lob['bids']['lob']) < n):
            r_n = 0
        else:
            r_n = lob['bids']['lob'][n - 1][1]

        if (len(lob['asks']['lob']) < n):
            q_n = 0
        else:
            q_n = lob['asks']['lob'][n - 1][1]
        return (r_n + q_n) / 2

    def set_opinion(self, updated_opinion):

        validated_update = updated_opinion

        if updated_opinion >= self.upper_op_bound:
            # set to upper bound
            validated_update = self.upper_op_bound
        elif updated_opinion <= self.lower_op_bound:
            # set to lower bound
            validated_update = self.lower_op_bound

        self.opinion = validated_update


    def sigmoid(self, num):

        return 2 * (1/(1+math.exp((-10.5)*num)) - 0.5)

    def adjust_strat(self):

        # if there is an imbalance, use the weighted average

        if(self.theta != 0):
            strat_val = (self.theta * self.wmkt) + ((1.0-self.wmkt) * self.opinion)

        # if there is no imbalance, trader solely off of opinion

        else:
            strat_val = self.opinion

        return strat_val

    def get_offset(self, lob, countdown, m):
        # a list to store MLOFI from level 1 to level m
        mlofi_list = [0 for i in range(m)]

        # a list to store the cumulative depth from level 1 to level m
        cd_list = [0 for i in range(m)]

        # a list to store the average depth from level 1 to level m
        ad_list = []

        n = 1

        # calculate MLOFI in each level
        while len(self.es_list) >= n:
            for i in range(m):
                mlofi_list[i] += self.es_list[-n]['level' + str(i+1)]
            n += 1
            # consider at most last 10 events
            if n >= 11:
                break

        n = 1
        # aggregate depths in each level
        while len(self.ds_list) >= n:
            for i in range(m):
                cd_list[i] += self.ds_list[-n]['level' + str(i+1)]
            n += 1
            # consider at most last 10 events
            if n >= 11:
                break

        # calculate average depth in each level
        for i in range(m):
            temp = None
            if n == 1:
                temp = cd_list[i]+1
            else:
                temp = cd_list[i]/(n-1)+1
            ad_list.append(temp)

        c = 5
        decay = 0.8
        offset = 0

        # calculate offset
        for i in range(m):
            offset += int(mlofi_list[i]*c*pow(decay,i)/ ad_list[i])

        return offset


    def set_uncertainty(self, updated_uncertainty):

        validated_update = updated_uncertainty

        if updated_uncertainty >= self.upper_un_bound:
            # set to upper bound
            validated_update = self.upper_un_bound
        elif updated_uncertainty <= self.lower_un_bound:
            # set to lower bound
            validated_update = self.lower_un_bound

        self.uncertainty = validated_update

    def calc_es(self, lob, m, verbose):
        new_e = {}
        for i in range(1, m + 1):
            new_e['level' + str(i)] = self.calc_level_n_e(lob, i)

        self.es_list.append(new_e)

    def getorder(self, time, countdown, lob):

        ####################################
        ## functioncs for PMF composition ##
        ####################################

        def P_join(left, right):

            if(type(left) == str):
                left = float(left)

            if(type(right) == str):
                right = float(right)

            # if left and/or right is a number (i.e., strat value or random constant),
            # then convert it into a pmf, then pass both into the parallel join function

            if(isinstance(left, float) or isinstance(left, int)):

                left = prob_mass_fn(left, None, None, 'null', pmin=self.updated_pmin, pmax=self.updated_pmax, dirn=self.current_trader_dir)

            if(isinstance(right, float) or isinstance(right, int)):

                right = prob_mass_fn(right, None, None, 'null', pmin=self.updated_pmin, pmax=self.updated_pmax, dirn=self.current_trader_dir)

            new_pmf =  prob_mass_fn(0.0, left, right, 'parallel', pmin=self.updated_pmin, pmax=self.updated_pmax, dirn=self.current_trader_dir)
            return new_pmf

        def S_join(left, right):

            if(type(left) == str):
                left = float(left)

            if(type(right) == str):
                right = float(right)

            # if left and/or right is a number (i.e., strat value or random constant),
            # then convert it into a pmf, then pass both into the series join function

            if(isinstance(left, float) or isinstance(left, int)):

                left = prob_mass_fn(left, None, None, 'null', pmin=self.updated_pmin, pmax=self.updated_pmax, dirn=self.current_trader_dir)

            if(isinstance(right, float) or isinstance(right, int)):

                right = prob_mass_fn(right, None, None, 'null', pmin=self.updated_pmin, pmax=self.updated_pmax, dirn=self.current_trader_dir)

            new_pmf = prob_mass_fn(0.5, left, right, 'series', pmin=self.updated_pmin, pmax=self.updated_pmax, dirn=self.current_trader_dir)
            return new_pmf


        def D_join(left, right):

            # if left and/or right is a number (i.e., strat value or random constant),
            # then convert it into a pmf, then pass both into the series join function

            if(type(left) == str):
                left = float(left)

            if(type(right) == str):
                right = float(right)

            if(isinstance(left, float) or isinstance(left, int)):

                left = prob_mass_fn(left, None, None, 'null', pmin=self.updated_pmin, pmax=self.updated_pmax, dirn=self.current_trader_dir)

            if(isinstance(right, float) or isinstance(right, int)):

                right = prob_mass_fn(right, None, None, 'null', pmin=self.updated_pmin, pmax=self.updated_pmax, dirn=self.current_trader_dir)

            new_pmf = prob_mass_fn(0.5, left, right, 'difference', pmin=self.updated_pmin, pmax=self.updated_pmax, dirn=self.current_trader_dir)
            return new_pmf

        # these functions all make the particular pmf have a different weight
        def weight_1(pmf):

            if(type(pmf) == str):
                pmf = float(pmf)

            # if pmf is a number, convert it to a pmf

            if(isinstance(pmf, float) or isinstance(pmf,int)):

                pmf = prob_mass_fn(pmf, None, None, 'null', pmin=self.updated_pmin, pmax=self.updated_pmax, dirn=self.current_trader_dir)

            pmf.assign_weight(1)
            return pmf

        def weight_2(pmf):

            if(type(pmf) == str):
                pmf = float(pmf)

            # if pmf is a number, convert it to a pmf

            if(isinstance(pmf, float) or isinstance(pmf,int)):

                pmf = prob_mass_fn(pmf, None, None, 'null', pmin=self.updated_pmin, pmax=self.updated_pmax, dirn=self.current_trader_dir)

            pmf.assign_weight(2)
            return pmf

        def weight_3(pmf):

            if(type(pmf) == str):
                pmf = float(pmf)

            # if pmf is a number, convert it to a pmf

            if(isinstance(pmf, float) or isinstance(pmf,int)):

                pmf = prob_mass_fn(pmf, None, None, 'null', pmin=self.updated_pmin, pmax=self.updated_pmax, dirn=self.current_trader_dir)

            pmf.assign_weight(3)
            return pmf

        def weight_4(pmf):

            if(type(pmf) == str):
                pmf = float(pmf)

            # if pmf is a number, convert it to a pmf

            if(isinstance(pmf, float) or isinstance(pmf,int)):

                pmf = prob_mass_fn(pmf, None, None, 'null', pmin=self.updated_pmin, pmax=self.updated_pmax, dirn=self.current_trader_dir)

            pmf.assign_weight(4)
            return pmf

        # shvr_price tells us what price a SHVR would quote in these circs
        def shvr_price(otype, limit, lob):

            if otype == 'Bid':
                if lob['bids']['n'] > 0:
                    shvr_p = lob['bids']['best'] + 1   # BSE tick size is always 1
                    if shvr_p > limit:
                        shvr_p = limit
                else:
                    shvr_p = lob['bids']['worst']
            else:
                if lob['asks']['n'] > 0:
                    shvr_p = lob['asks']['best'] - 1   # BSE tick size is always 1
                    if shvr_p < limit:
                        shvr_p = limit
                else:
                    shvr_p = lob['asks']['worst']

            return shvr_p

        def calc_cdf_lut(strat, dirn, pmin, pmax):

            # at the moment, can call our pmf spec func to give us a pmf table
            # need to convert that to cdf

            p_range = float(pmax - pmin)
            if p_range < 1:
                # special case: the SHVR-style strategy has shaved all the way to the limit price
                # the lower and upper bounds on the interval are adjacent prices;
                # so cdf is simply the lower price with probability 1

                cdf=[]
                cdf.append({'price':pmin, 'prob': 1.0, 'cum_prob': 1.0, 'cal_p': 0, 'cal_sum': 0})

                return {'strat':strat, 'dirn':dirn, 'pmin':pmin, 'pmax':pmax, 'cdf_lut':cdf}

            if(dirn == -1):
                ARG0=self.strat
                # final result from buyer GP simulations
                pmf_table = P_join(P_join(ARG0, ARG0),ARG0)

            if(dirn == 1):
                ARG0=self.strat
                # final result from seller GP simulations
                pmf_table = S_join(P_join(S_join(-0.6370401285276484, ARG0), ARG0), -0.6179344894946064)

            pmf_table = pmf_table.get_pmf()
            return pmf_table

        verbose = False

        if verbose:
            print('PRZI getorder: strat=%f' % self.strat)

        if len(self.orders) < 1:
            # no orders: return NULL
            order = None
        else:

            # unpack the assignment-order
            limit = self.orders[0].price
            self.updated_pmax = int(limit)
            otype = self.orders[0].otype

            microp = lob['microprice']
            midp = lob['midprice']

            # get extreme limits on price interval
            # lowest price the market will bear
            minprice = int(lob['bids']['worst'])  # default assumption: worst bid price possible is 1 tick
            # trader's individual estimate highest price the market will bear
            if self.pmax is None:
                maxprice = int(limit * self.pmax_c_i + 0.5) # in the absence of any other info, guess
                self.pmax = maxprice
            elif lob['asks']['sess_hi'] is not None:
                if self.pmax < lob['asks']['sess_hi']:        # some other trader has quoted higher than I expected
                    maxprice = lob['asks']['sess_hi']           # so use that as my new estimate of highest
                    self.pmax = maxprice
                else:
                    maxprice = self.pmax
            else:
                maxprice = self.pmax

            # what price would a SHVR quote?
            p_shvr = shvr_price(otype, limit, lob)

            # before we do anything, need to update opinion based on market
            # here is where we calculate offset

            offset = self.get_offset(lob, countdown, self.m)

            # sometimes microprice doesn't exist
            # in that case, use pmax to normalise, else use microprice
            try:
                offset /= lob['microprice']
            except:
                offset /= self.pmax

            # put through sigmoid function, result is theta
            self.theta = self.sigmoid(offset)

            # calculate new strat value
            self.strat = self.adjust_strat()


            # if this trader is an intelligent extremist, they will just set
            # their composite opinion and strategy equal to the opinion tendency value

            if(self.intExtremistValue is not None):
                if(self.intExtremistValue > 0):
                    k = 0.002
                    self.opinion = min(1,(k*(time-self.time_activated)))
                    self.strat = self.opinion
                if(self.intExtremistValue < 0):
                    k = -0.002
                    self.opinion = max(-1,(k*(time-self.time_activated)))
                    self.strat = self.opinion

            # to reduce the impact of a largely decreasing pmax as s approaches 1
            # and a largely increasing pmin as s approaches -1, evaluate to 1
            if(self.strat >= 0.99):
                self.strat = 1
            if(self.strat <= -0.99):
                self.strat = -1

            # update the "updated" values for use in PMF joins
            self.opinion = self.strat

            # the LUTs are re-computed if any of the details have changed
            if otype == 'Bid':

                self.current_trader_dir = -1

                # direction * strat
                dxs = -1 * self.strat  # the minus one multiplier is the "buy" direction

                p_max = int(limit)

                if dxs <= 0:
                    p_min = minprice        # this is delta_p for BSE, i.e. ticksize =1

                else:
                    # shade the lower bound on the interval
                    # away from minprice and toward shvr_price
                    p_min = int(0.5 + (dxs * p_shvr) + ((1.0-dxs) * minprice))

                if(dxs == -1):
                    p_min = p_max           # if pure-gvwy strategy, set p_min = p_max

                if(dxs == 1):
                    p_max = p_min       # if pure-shvr strategy, set p_max = p_max = shvr price

                self.updated_pmin = p_min
                self.updated_pmax = p_max

                if (self.cdf_lut_bid is None) or \
                        (self.cdf_lut_bid['strat'] != self.strat) or\
                        (self.cdf_lut_bid['pmin'] != p_min) or \
                        (self.cdf_lut_bid['pmax'] != p_max):

                    # need to compute a new LUT
                    if verbose:
                        print('New bid LUT')
                    self.cdf_lut_bid = calc_cdf_lut(self.strat, self.current_trader_dir, p_min, p_max)

                lut = self.cdf_lut_bid

            else:   # otype == 'Ask'

                self.current_trader_dir = 1

                dxs = self.strat

                p_min = int(limit)

                if dxs <= 0:
                    p_max = maxprice
                else:
                    # shade the upper bound on the interval
                    # away from maxprice and toward shvr_price
                    p_max = int(0.5 + (dxs * p_shvr) + ((1.0-dxs) * maxprice))

                if(dxs == -1):
                    p_min = p_max           # if pure-gvwy strategy, set p_min = p_max

                if(dxs == 1):
                    p_max = p_min       # if pure-shvr strategy, set p_max = p_max = shvr price

                self.updated_pmin = p_min
                self.updated_pmax = p_max

                if (self.cdf_lut_ask is None) or \
                        (self.cdf_lut_ask['strat'] != self.strat) or \
                        (self.cdf_lut_ask['pmin'] != p_min) or \
                        (self.cdf_lut_ask['pmax'] != p_max):
                    # need to compute a new LUT
                    if verbose:
                        print('New ask LUT')
                    self.cdf_lut_ask = calc_cdf_lut(self.strat, self.current_trader_dir, p_min, p_max)


                lut = self.cdf_lut_ask

            if verbose:
                print('PRZI LUT =', lut)

            u = random.random()

            quoteprice = None

            for entry in lut['cdf_lut']:
                if u < entry['cum_prob']:
                    quoteprice = entry['price']
                    break

            if(lut is None or quoteprice is None):
                return None

            order = Order(self.tid, otype,
                          quoteprice, self.orders[0].qty, time, lob['QID'])

            self.lastquote = order


            return order
