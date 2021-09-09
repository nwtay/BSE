# -*- coding: utf-8 -*-
#
# BSE: The Bristol Stock Exchange
#
# Version 1.4; 26 Oct 2020 (Python 3.x)
# Version 1.3; July 21st, 2018 (Python 2.x)
# Version 1.2; November 17th, 2012 (Python 2.x)
#
# Copyright (c) 2012-2020, Dave Cliff
#
#
# ------------------------
#
# MIT Open-Source License:
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ------------------------
#
#
#
# BSE is a very simple simulation of automated execution traders
# operating on a very simple model of a limit order book (LOB) exchange
#
# major simplifications in this version:
#       (a) only one financial instrument being traded
#       (b) traders can only trade contracts of size 1 (will add variable quantities later)
#       (c) each trader can have max of one order per single orderbook.
#       (d) traders can replace/overwrite earlier orders, and/or can cancel
#       (d) simply processes each order in sequence and republishes LOB to all traders
#           => no issues with exchange processing latency/delays or simultaneously issued orders.
#
# NB this code has been written to be readable/intelligible, not efficient!

# could import pylab here for graphing etc

import sys
import operator
import math
import numpy
import random
import pandas as pd
import matplotlib.pyplot as plt
import PMF_joins
from traders import Order, Trader, Trader_Giveaway, Trader_Shaver, Trader_Sniper, Trader_ZIC, Trader_ZIP, Trader_opinionated_ZIC, Trader_OPRZI, Trader_PRZI
from op_models import bounded_confidence_step, relative_agreement_step, relative_disagreement_step, relative_disagreement_step_mix
import csv
from datetime import datetime
import profile

from deap import gp
from deap import creator
from deap import base
from deap import tools
from deap import algorithms


bse_sys_minprice = 1  # minimum price in the system, in cents/pennies
bse_sys_maxprice = 500  # maximum price in the system, in cents/pennies
ticksize = 1  # minimum change in price, in cents/penniess

class prob_mass_fn:

    # here, generate a pmf based on a value (i.e., in calc_lut)

    def __init__(self, stratval, pmf1, pmf2, keyword, pmin, pmax, dirn):

        if(keyword == 'null'):
            self.pmf = create_pmf(stratval, 100, 4, dirn, pmin, pmax)
            self.weight = 1
            self.stratval = stratval

        if(keyword == 'parallel'):
            self.weight = 1
            self.pmf = PMF_joins.P_join(pmf1, pmf2)

        if(keyword == 'series'):
            self.weight = 1
            self.pmf = PMF_joins.S_join(pmf1, pmf2)

        if(keyword == 'difference'):
            self.weight = 1
            self.pmf = PMF_joins.D_join(pmf1, pmf2)

    def plot(self, color, fignum):
        df = pd.DataFrame(self.pmf)
        plt.figure(fignum)
        plt.plot(df['price'], df['prob'], 'k-', color=color)

    def assign_weight(self, weight):
        self.weight = weight

    def get_pmf(self):
        return self.pmf

    def get_weight(self):
        return self.weight

    def __call__(self):
        return self


# Orderbook_half is one side of the book: a list of bids or a list of asks, each sorted best-first

class Orderbook_half:

    def __init__(self, booktype, worstprice):
        # booktype: bids or asks?
        self.booktype = booktype
        # dictionary of orders received, indexed by Trader ID
        self.orders = {}
        # limit order book, dictionary indexed by price, with order info
        self.lob = {}
        # anonymized LOB, lists, with only price/qty info
        self.lob_anon = []
        # summary stats
        self.best_price = None
        self.best_tid = None
        self.worstprice = worstprice
        self.session_extreme = None    # most extreme price quoted in this session
        self.n_orders = 0  # how many orders?
        self.lob_depth = 0  # how many different prices on lob?

    def anonymize_lob(self):
        # anonymize a lob, strip out order details, format as a sorted list
        # NB for asks, the sorting should be reversed
        self.lob_anon = []
        for price in sorted(self.lob):
            qty = self.lob[price][0]
            self.lob_anon.append([price, qty])

    def build_lob(self):
        lob_verbose = False
        # take a list of orders and build a limit-order-book (lob) from it
        # NB the exchange needs to know arrival times and trader-id associated with each order
        # returns lob as a dictionary (i.e., unsorted)
        # also builds anonymized version (just price/quantity, sorted, as a list) for publishing to traders
        self.lob = {}
        for tid in self.orders:
            order = self.orders.get(tid)
            price = order.price
            if price in self.lob:
                # update existing entry
                qty = self.lob[price][0]
                orderlist = self.lob[price][1]
                orderlist.append([order.time, order.qty, order.tid, order.qid])
                self.lob[price] = [qty + order.qty, orderlist]
            else:
                # create a new dictionary entry
                self.lob[price] = [order.qty, [[order.time, order.qty, order.tid, order.qid]]]
        # create anonymized version
        self.anonymize_lob()
        # record best price and associated trader-id
        if len(self.lob) > 0:
            if self.booktype == 'Bid':
                self.best_price = self.lob_anon[-1][0]
            else:
                self.best_price = self.lob_anon[0][0]
            self.best_tid = self.lob[self.best_price][1][0][2]
        else:
            self.best_price = None
            self.best_tid = None

        if lob_verbose:
            print(self.lob)

    def book_add(self, order):
        # add order to the dictionary holding the list of orders
        # either overwrites old order from this trader
        # or dynamically creates new entry in the dictionary
        # so, max of one order per trader per list
        # checks whether length or order list has changed, to distinguish addition/overwrite
        # print('book_add > %s %s' % (order, self.orders))

        # if this is an ask, does the price set a new extreme-high record?
        if (self.booktype == 'Ask') and ((self.session_extreme is None) or (order.price > self.session_extreme)):
            self.session_extreme = int(order.price)

        n_orders = self.n_orders
        self.orders[order.tid] = order
        self.n_orders = len(self.orders)
        self.build_lob()
        # print('book_add < %s %s' % (order, self.orders))
        if n_orders != self.n_orders:
            return 'Addition'
        else:
            return 'Overwrite'

    def book_del(self, order):
        # delete order from the dictionary holding the orders
        # assumes max of one order per trader per list
        # checks that the Trader ID does actually exist in the dict before deletion
        # print('book_del %s',self.orders)
        if self.orders.get(order.tid) is not None:
            del (self.orders[order.tid])
            self.n_orders = len(self.orders)
            self.build_lob()
        # print('book_del %s', self.orders)

    def delete_best(self):
        # delete order: when the best bid/ask has been hit, delete it from the book
        # the TraderID of the deleted order is return-value, as counterparty to the trade
        best_price_orders = self.lob[self.best_price]
        best_price_qty = best_price_orders[0]
        best_price_counterparty = best_price_orders[1][0][2]
        if best_price_qty == 1:
            # here the order deletes the best price
            del (self.lob[self.best_price])
            del (self.orders[best_price_counterparty])
            self.n_orders = self.n_orders - 1
            if self.n_orders > 0:
                if self.booktype == 'Bid':
                    self.best_price = max(self.lob.keys())
                else:
                    self.best_price = min(self.lob.keys())
                self.lob_depth = len(self.lob.keys())
            else:
                self.best_price = self.worstprice
                self.lob_depth = 0
        else:
            # best_bid_qty>1 so the order decrements the quantity of the best bid
            # update the lob with the decremented order data
            self.lob[self.best_price] = [best_price_qty - 1, best_price_orders[1][1:]]

            # update the bid list: counterparty's bid has been deleted
            del (self.orders[best_price_counterparty])
            self.n_orders = self.n_orders - 1
        self.build_lob()
        return best_price_counterparty




# Orderbook for a single instrument: list of bids and list of asks

class Orderbook(Orderbook_half):

    def __init__(self):
        self.bids = Orderbook_half('Bid', bse_sys_minprice)
        self.asks = Orderbook_half('Ask', bse_sys_maxprice)
        self.tape = []
        self.quote_id = 0  # unique ID code for each quote accepted onto the book

    def midprice(self, bid_p, bid_q, ask_p, ask_q):
            # returns midprice as mean of best bid and best ask if both best bid & best ask exist
            # if only one best price exists, returns that as mid
            # if neither best price exists, returns None
            mprice = None
            if bid_q > 0 and ask_q == None :
                    mprice = bid_p
            elif ask_q > 0 and bid_q == None :
                    mprice = ask_p
            elif bid_q>0 and ask_q >0 :
                    mprice = ( bid_p + ask_p ) / 2.0
            return mprice


    def microprice(self, bid_p, bid_q, ask_p, ask_q):
            mprice = None
            if bid_q>0 and ask_q >0 :
                    tot_q = bid_q + ask_q
                    mprice = ( (bid_p * ask_q) + (ask_p * bid_q) ) / tot_q
            return mprice


# Exchange's internal orderbook

class Exchange(Orderbook):

    def add_order(self, order, verbose):
        # add a quote/order to the exchange and update all internal records; return unique i.d.
        order.qid = self.quote_id
        self.quote_id = order.qid + 1
        # if verbose : print('QUID: order.quid=%d self.quote.id=%d' % (order.qid, self.quote_id))
        if order.otype == 'Bid':
            response = self.bids.book_add(order)
            best_price = self.bids.lob_anon[-1][0]
            self.bids.best_price = best_price
            self.bids.best_tid = self.bids.lob[best_price][1][0][2]
        else:
            response = self.asks.book_add(order)
            best_price = self.asks.lob_anon[0][0]
            self.asks.best_price = best_price
            self.asks.best_tid = self.asks.lob[best_price][1][0][2]
        return [order.qid, response]

    def del_order(self, time, order, verbose):
        # delete a trader's quot/order from the exchange, update all internal records
        if order.otype == 'Bid':
            self.bids.book_del(order)
            if self.bids.n_orders > 0:
                best_price = self.bids.lob_anon[-1][0]
                self.bids.best_price = best_price
                self.bids.best_tid = self.bids.lob[best_price][1][0][2]
            else:  # this side of book is empty
                self.bids.best_price = None
                self.bids.best_tid = None
            cancel_record = {'type': 'Cancel', 'time': time, 'order': order}
            self.tape.append(cancel_record)

        elif order.otype == 'Ask':
            self.asks.book_del(order)
            if self.asks.n_orders > 0:
                best_price = self.asks.lob_anon[0][0]
                self.asks.best_price = best_price
                self.asks.best_tid = self.asks.lob[best_price][1][0][2]
            else:  # this side of book is empty
                self.asks.best_price = None
                self.asks.best_tid = None
            cancel_record = {'type': 'Cancel', 'time': time, 'order': order}
            self.tape.append(cancel_record)
        else:
            # neither bid nor ask?
            sys.exit('bad order type in del_quote()')

    def process_order2(self, time, order, verbose):
        # receive an order and either add it to the relevant LOB (ie treat as limit order)
        # or if it crosses the best counterparty offer, execute it (treat as a market order)
        oprice = order.price
        counterparty = None
        [qid, response] = self.add_order(order, verbose)  # add it to the order lists -- overwriting any previous order
        order.qid = qid
        if verbose:
            print('QUID: order.quid=%d' % order.qid)
            print('RESPONSE: %s' % response)
        best_ask = self.asks.best_price
        best_ask_tid = self.asks.best_tid
        best_bid = self.bids.best_price
        best_bid_tid = self.bids.best_tid
        if order.otype == 'Bid':
            if self.asks.n_orders > 0 and best_bid >= best_ask:
                # bid lifts the best ask
                if verbose:
                    print("Bid $%s lifts best ask" % oprice)
                counterparty = best_ask_tid
                price = best_ask  # bid crossed ask, so use ask price
                if verbose:
                    print('counterparty, price', counterparty, price)
                # delete the ask just crossed
                self.asks.delete_best()
                # delete the bid that was the latest order
                self.bids.delete_best()
        elif order.otype == 'Ask':
            if self.bids.n_orders > 0 and best_ask <= best_bid:
                # ask hits the best bid
                if verbose:
                    print("Ask $%s hits best bid" % oprice)
                # remove the best bid
                counterparty = best_bid_tid
                price = best_bid  # ask crossed bid, so use bid price
                if verbose:
                    print('counterparty, price', counterparty, price)
                # delete the bid just crossed, from the exchange's records
                self.bids.delete_best()
                # delete the ask that was the latest order, from the exchange's records
                self.asks.delete_best()
        else:
            # we should never get here
            sys.exit('process_order() given neither Bid nor Ask')
        # NB at this point we have deleted the order from the exchange's records
        # but the two traders concerned still have to be notified
        if verbose:
            print('counterparty %s' % counterparty)
        if counterparty is not None:
            # process the trade
            if verbose: print('>>>>>>>>>>>>>>>>>TRADE t=%010.3f $%d %s %s' % (time, price, counterparty, order.tid))
            transaction_record = {'type': 'Trade',
                                  'time': time,
                                  'price': price,
                                  'party1': counterparty,
                                  'party2': order.tid,
                                  'qty': order.qty
                                  }
            self.tape.append(transaction_record)
            return transaction_record
        else:
            return None

    # Currently tape_dump only writes a list of transactions (ignores cancellations)
    def tape_dump(self, fname, fmode, tmode):
        dumpfile = open(fname, fmode)
        for tapeitem in self.tape:
            if tapeitem['type'] == 'Trade':
                dumpfile.write('%010.3f, %s\n' % (tapeitem['time'], tapeitem['price']))
        dumpfile.close()
        if tmode == 'wipe':
            self.tape = []

    # this returns the LOB data "published" by the exchange,
    # i.e., what is accessible to the traders
    def publish_lob(self, time, verbose):

        n_bids = len(self.bids.orders)
        n_asks = len(self.asks.orders)

        public_data = {}
        public_data['time'] = time
        public_data['bids'] = {'best': self.bids.best_price,
                               'worst': self.bids.worstprice,
                               'n': self.bids.n_orders,
                               'lob': self.bids.lob_anon}
        public_data['asks'] = {'best': self.asks.best_price,
                               'worst': self.asks.worstprice,
                               'sess_hi': self.asks.session_extreme,
                               'n': self.asks.n_orders,
                               'lob': self.asks.lob_anon}
        public_data['QID'] = self.quote_id
        public_data['tape'] = self.tape
        if verbose:
            print('publish_lob: t=%d' % time)
            print('BID_lob=%s' % public_data['bids']['lob'])
            # print('best=%s; worst=%s; n=%s ' % (self.bids.best_price, self.bids.worstprice, self.bids.n_orders))
            print('ASK_lob=%s' % public_data['asks']['lob'])
            # print('qid=%d' % self.quote_id)

        public_data['midprice'] = None
        public_data['microprice'] = None
        if n_bids>0 and n_asks>0 :
                # neither side of the LOB is empty
                best_bid_q= self.bids.lob_anon[0][1]
                best_ask_q = self.asks.lob_anon[0][1]
                public_data['midprice'] = self.midprice(self.bids.best_price, best_bid_q, self.asks.best_price, best_ask_q)
                public_data['microprice'] = self.microprice(self.bids.best_price, best_bid_q, self.asks.best_price, best_ask_q)

        return public_data

####################################################################################
# BSE_GP trader defined here

class Trader_OPRZI_GP(Trader):

    def __init__(self, ttype, tid, balance, time, m, opinion, uncertainty, lower_op_bound, upper_op_bound, start_opinion):
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

        self.wmkt = self.init_wmkt()

    def add_pmf_spec(self, pmf_spec):

        self.pmf_spec = pmf_spec
        #print(updated_strat)
        #result.plot('green', 1)
        #plt.show()

    ##################################
    ## Functions needed to respond ###
    ##################################

    def respond(self, time, lob, trade, verbose):

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

    def init_wmkt(self):

        # first random draw (are you 30% of the population?)
        return 0.5

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

            pmf_table = pmf_spec(ARG0=updated_strat)

            # in case the individual is just a float
            if(isinstance(pmf_table, float)):
                pmf_table = prob_mass_fn(pmf_table, None, None, 'null', pmin=updated_pmin, pmax=updated_pmax, dirn=current_trader_dir)

            pmf_table = pmf_table.get_pmf()
            return pmf_table

        # update the global vars
        global updated_pmax
        global updated_pmin
        global current_trader_dir
        global updated_strat

        verbose = False

        if verbose:
            print('PRZI getorder: strat=%f' % self.strat)

        if len(self.orders) < 1:
            # no orders: return NULL
            order = None
        else:

            #updated_pmin = 1
            #updated_pmax = 200
            #current_trader_dir = -1
            #updated_strat = 0.0

            # unpack the assignment-order
            limit = self.orders[0].price
            updated_pmax = int(limit)
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


            # before we do anything, need to update opinion based on market
            # HERE IS WHERE WE CALCAULTE OFFSET!!

            offset = self.get_offset(lob, countdown, self.m)

            # sometimes microprice doesn't exist
            # in that case, use pmax to normalise, else use microprice
            try:
                offset /= lob['microprice']
            except:
                offset /= self.pmax

            # put through sigmoid function
            self.theta = self.sigmoid(offset)

            # calculate new strat value
            self.strat = self.adjust_strat()

            updated_strat = self.strat

            # the LUTs are re-computed if any of the details have changed
            if otype == 'Bid':

                current_trader_dir = -1

                # direction * strat
                dxs = -1 * self.strat  # the minus one multiplier is the "buy" direction

                p_max = int(limit)
                updated_pmax = p_max

                if dxs <= 0:
                    p_min = minprice        # this is delta_p for BSE, i.e. ticksize =1

                else:
                    # shade the lower bound on the interval
                    # away from minprice and toward shvr_price
                    p_min = int(0.5 + (dxs * p_shvr) + ((1.0-dxs) * minprice))

                updated_pmin = p_min

                if (self.cdf_lut_bid is None) or \
                        (self.cdf_lut_bid['strat'] != self.strat) or\
                        (self.cdf_lut_bid['pmin'] != p_min) or \
                        (self.cdf_lut_bid['pmax'] != p_max):
                    # need to compute a new LUT
                    if verbose:
                        print('New bid LUT')
                    self.cdf_lut_bid = calc_cdf_lut(self.strat, current_trader_dir, updated_pmin, updated_pmax)

                lut = self.cdf_lut_bid

            else:   # otype == 'Ask'

                current_trader_dir = 1

                dxs = self.strat

                p_min = int(limit)
                updated_pmin = p_min
                if dxs <= 0:
                    p_max = maxprice

                else:
                    # shade the upper bound on the interval
                    # away from maxprice and toward shvr_price
                    p_max = int(0.5 + (dxs * p_shvr) + ((1.0-dxs) * maxprice))

                updated_pmax = p_max

                if (self.cdf_lut_ask is None) or \
                        (self.cdf_lut_ask['strat'] != self.strat) or \
                        (self.cdf_lut_ask['pmin'] != p_min) or \
                        (self.cdf_lut_ask['pmax'] != p_max):
                    # need to compute a new LUT
                    if verbose:
                        print('New ask LUT')
                    self.cdf_lut_ask = calc_cdf_lut(self.strat, current_trader_dir, updated_pmin, updated_pmax)


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


##########################---Below lies the experiment/test-rig---##################


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


# trade_stats()
# dump CSV statistics on exchange data and trader population to file for later analysis
# this makes no assumptions about the number of types of traders, or
# the number of traders of any one type -- allows either/both to change
# between successive calls, but that does make it inefficient as it has to
# re-analyse the entire set of traders on each call
def trade_stats(expid, traders, dumpfile, time, lob):

    # Analyse the set of traders, to see what types we have
    trader_types = {}
    for t in traders:
        ttype = traders[t].ttype
        if ttype in trader_types.keys():
            t_balance = trader_types[ttype]['balance_sum'] + traders[t].balance
            n = trader_types[ttype]['n'] + 1
        else:
            t_balance = traders[t].balance
            n = 1
        trader_types[ttype] = {'n': n, 'balance_sum': t_balance}

    # first two columns of output are the session_id and the time
    dumpfile.write('%s, %06d, ' % (expid, time))

    # second two columns of output are the LOB best bid and best offer (or 'None' if they're undefined)
    if lob['bids']['best'] is not None:
        dumpfile.write('%d, ' % (lob['bids']['best']))
    else:
        dumpfile.write('None, ')
    if lob['asks']['best'] is not None:
        dumpfile.write('%d, ' % (lob['asks']['best']))
    else:
        dumpfile.write('None, ')

    # total remaining number of columns printed depends on number of different trader-types at this timestep
    # for each trader type we print FOUR columns...
    # TraderTypeCode, TotalProfitForThisTraderType, NumberOfTradersOfThisType, AverageProfitPerTraderOfThisType
    for ttype in sorted(list(trader_types.keys())):
        n = trader_types[ttype]['n']
        s = trader_types[ttype]['balance_sum']
        dumpfile.write('%s, %d, %d, %f, ' % (ttype, s, n, s / float(n)))

    if lob['bids']['best'] is not None:
        dumpfile.write('%d, ' % (lob['bids']['best']))
    else:
        dumpfile.write('N, ')
    if lob['asks']['best'] is not None:
        dumpfile.write('%d, ' % (lob['asks']['best']))
    else:
        dumpfile.write('N, ')

    dumpfile.write('\n')


# create a bunch of traders from traders_spec
# returns tuple (n_buyers, n_sellers)
# optionally shuffles the pack of buyers and the pack of sellers
# also initialises opinions of traders
def populate_market(traders_spec, traders, shuffle, verbose, model):
    def trader_type(robottype, name, Min_Op, Max_Op, u_min, u_max, model, start_opinion, extreme_distance):
        opinion = 0.5
        uncertainty = 1.0
        if model == 'BC':
            opinion = random.uniform(Min_Op+extreme_distance, Max_Op-extreme_distance)
        elif model == 'RA':
            opinion = random.uniform(Min_Op+extreme_distance, Max_Op-extreme_distance)
            uncertainty = random.uniform(u_min, u_max)
        elif model == 'RD':
            opinion = random.uniform(Min_Op+extreme_distance, Max_Op-extreme_distance)
            uncertainty = random.uniform(u_min, u_max)
            # uncertainty = min((random.uniform(0.2, 2.0) + random.uniform(0, 1)), 2)
        else:
            sys.exit('FATAL: don\'t know that opinion dynamic model type %s\n' % model);

        if robottype == 'GVWY':
                return Trader_Giveaway('GVWY', name, 0.00, 0, opinion, uncertainty, Min_Op, Max_Op, start_opinion)
        elif robottype == 'ZIC':
                return Trader_ZIC('ZIC', name, 0.00, 0, opinion, uncertainty, Min_Op, Max_Op, start_opinion)
        elif robottype == 'SHVR':
                return Trader_Shaver('SHVR', name, 0.00, 0, opinion, uncertainty, Min_Op, Max_Op, start_opinion)
        elif robottype == 'SNPR':
                return Trader_Sniper('SNPR', name, 0.00, 0, opinion, uncertainty, Min_Op, Max_Op, start_opinion)
        elif robottype == 'ZIP':
                return Trader_ZIP('ZIP', name, 0.00, 0, opinion, uncertainty, Min_Op, Max_Op, start_opinion)
        elif robottype == 'O-ZIC':
                return Trader_opinionated_ZIC('O-ZIC', name, 0.00, 0, opinion, uncertainty, Min_Op, Max_Op, start_opinion)
        elif robottype == 'PRZI':
                return Trader_PRZI('PRZI', name, 0.00, 0, opinion, uncertainty, Min_Op, Max_Op, start_opinion)
        elif robottype == 'O-PRZI':
                return Trader_OPRZI('O-PRZI', name, 0.00, 0, 3, opinion, uncertainty, Min_Op, Max_Op, start_opinion)
        elif robottype == 'O-PRZI-GP':
                new_trader = Trader_OPRZI_GP('O-PRZI-GP', name, 0.00, 0, 3, opinion, uncertainty, Min_Op, Max_Op, start_opinion)
                new_trader.add_pmf_spec(pmf_spec)
                return new_trader
        else:
                sys.exit('FATAL: don\'t know robot type %s\n' % robottype)

    def shuffle_traders(ttype_char, n, traders):
        for swap in range(n):
            t1 = (n - 1) - swap
            t2 = random.randint(0, t1)
            t1name = '%c%02d' % (ttype_char, t1)
            t2name = '%c%02d' % (ttype_char, t2)
            traders[t1name].tid = t2name
            traders[t2name].tid = t1name
            temp = traders[t1name]
            traders[t1name] = traders[t2name]
            traders[t2name] = temp

    n_buyers = 0
    for bs in traders_spec['buyers']:
        ttype = bs[0]
        for b in range(bs[1]):
            tname = 'B%02d' % n_buyers  # buyer i.d. string
            traders[tname] = trader_type(ttype, tname, Min_Op, Max_Op, u_min, u_max, model, "moderate", extreme_distance)
            n_buyers = n_buyers + 1

    if n_buyers < 1:
        sys.exit('FATAL: no buyers specified\n')

    if shuffle:
        shuffle_traders('B', n_buyers, traders)

    n_sellers = 0
    for ss in traders_spec['sellers']:
        ttype = ss[0]
        for s in range(ss[1]):
            tname = 'S%02d' % n_sellers  # buyer i.d. string
            traders[tname] = trader_type(ttype, tname, Min_Op, Max_Op, u_min, u_max, model, "moderate", extreme_distance)
            n_sellers = n_sellers + 1

    if n_sellers < 1:
        sys.exit('FATAL: no sellers specified\n')

    if shuffle:
        shuffle_traders('S', n_sellers, traders)

    verbose = False
    if verbose:
        for t in range(n_buyers):
            bname = 'B%02d' % t
            print(traders[bname])
        for t in range(n_sellers):
            bname = 'S%02d' % t
            print(traders[bname])

    return {'n_buyers': n_buyers, 'n_sellers': n_sellers}


# customer_orders(): allocate orders to traders
# parameter "os" is order schedule
# os['timemode'] is either 'periodic', 'drip-fixed', 'drip-jitter', or 'drip-poisson'
# os['interval'] is number of seconds for a full cycle of replenishment
# drip-poisson sequences will be normalised to ensure time of last replenishment <= interval
# parameter "pending" is the list of future orders (if this is empty, generates a new one from os)
# revised "pending" is the returned value
#
# also returns a list of "cancellations": trader-ids for those traders who are now working a new order and hence
# need to kill quotes already on LOB from working previous order
#
#
# if a supply or demand schedule mode is "random" and more than one range is supplied in ranges[],
# then each time a price is generated one of the ranges is chosen equiprobably and
# the price is then generated uniform-randomly from that range
#
# if len(range)==2, interpreted as min and max values on the schedule, specifying linear supply/demand curve
# if len(range)==3, first two vals are min & max, third value should be a function that generates a dynamic price offset
#                   -- the offset value applies equally to the min & max, so gradient of linear sup/dem curve doesn't vary
# if len(range)==4, the third value is function that gives dynamic offset for schedule min,
#                   and fourth is a function giving dynamic offset for schedule max, so gradient of sup/dem linear curve can vary
#
# the interface on this is a bit of a mess... could do with refactoring


def customer_orders(time, last_update, traders, trader_stats, os, pending, verbose):
    def sysmin_check(price):
        if price < bse_sys_minprice:
            print('WARNING: price < bse_sys_min -- clipped')
            price = bse_sys_minprice
        return price

    def sysmax_check(price):
        if price > bse_sys_maxprice:
            print('WARNING: price > bse_sys_max -- clipped')
            price = bse_sys_maxprice
        return price

    def getorderprice(i, sched, n, mode, issuetime):
        # does the first schedule range include optional dynamic offset function(s)?
        if len(sched[0]) > 2:
            offsetfn = sched[0][2]
            if callable(offsetfn):
                # same offset for min and max
                offset_min = offsetfn(issuetime)
                offset_max = offset_min
            else:
                sys.exit('FAIL: 3rd argument of sched in getorderprice() not callable')
            if len(sched[0]) > 3:
                # if second offset function is specfied, that applies only to the max value
                offsetfn = sched[0][3]
                if callable(offsetfn):
                    # this function applies to max
                    offset_max = offsetfn(issuetime)
                else:
                    sys.exit('FAIL: 4th argument of sched in getorderprice() not callable')
        else:
            offset_min = 0.0
            offset_max = 0.0

        pmin = sysmin_check(offset_min + min(sched[0][0], sched[0][1]))
        pmax = sysmax_check(offset_max + max(sched[0][0], sched[0][1]))
        prange = pmax - pmin
        stepsize = prange / (n - 1)
        halfstep = round(stepsize / 2.0)

        if mode == 'fixed':
            orderprice = pmin + int(i * stepsize)
        elif mode == 'jittered':
            orderprice = pmin + int(i * stepsize) + random.randint(-halfstep, halfstep)
        elif mode == 'random':
            if len(sched) > 1:
                # more than one schedule: choose one equiprobably
                s = random.randint(0, len(sched) - 1)
                pmin = sysmin_check(min(sched[s][0], sched[s][1]))
                pmax = sysmax_check(max(sched[s][0], sched[s][1]))
            orderprice = random.randint(pmin, pmax)
        else:
            sys.exit('FAIL: Unknown mode in schedule')
        orderprice = sysmin_check(sysmax_check(orderprice))
        return orderprice

    def getissuetimes(n_traders, mode, interval, shuffle, fittointerval):
        interval = float(interval)
        if n_traders < 1:
            sys.exit('FAIL: n_traders < 1 in getissuetime()')
        elif n_traders == 1:
            tstep = interval
        else:
            tstep = interval / (n_traders - 1)
        arrtime = 0
        issuetimes = []
        for t in range(n_traders):
            if mode == 'periodic':
                arrtime = interval
            elif mode == 'drip-fixed':
                arrtime = t * tstep
            elif mode == 'drip-jitter':
                arrtime = t * tstep + tstep * random.random()
            elif mode == 'drip-poisson':
                # poisson requires a bit of extra work
                interarrivaltime = random.expovariate(n_traders / interval)
                arrtime += interarrivaltime
            else:
                sys.exit('FAIL: unknown time-mode in getissuetimes()')
            issuetimes.append(arrtime)

            # at this point, arrtime is the last arrival time
        if fittointerval and ((arrtime > interval) or (arrtime < interval)):
            # generated sum of interarrival times longer than the interval
            # squish them back so that last arrival falls at t=interval
            for t in range(n_traders):
                issuetimes[t] = interval * (issuetimes[t] / arrtime)
        # optionally randomly shuffle the times
        if shuffle:
            for t in range(n_traders):
                i = (n_traders - 1) - t
                j = random.randint(0, i)
                tmp = issuetimes[i]
                issuetimes[i] = issuetimes[j]
                issuetimes[j] = tmp
        return issuetimes

    def getschedmode(time, os):
        got_one = False
        for sched in os:
            if (sched['from'] <= time) and (time < sched['to']):
                # within the timezone for this schedule
                schedrange = sched['ranges']
                mode = sched['stepmode']
                got_one = True
                exit  # jump out the loop -- so the first matching timezone has priority over any others
        if not got_one:
            sys.exit('Fail: time=%5.2f not within any timezone in os=%s' % (time, os))
        return (schedrange, mode)

    n_buyers = trader_stats['n_buyers']
    n_sellers = trader_stats['n_sellers']

    shuffle_times = True

    cancellations = []

    if len(pending) < 1:
        # list of pending (to-be-issued) customer orders is empty, so generate a new one
        new_pending = []

        # demand side (buyers)
        issuetimes = getissuetimes(n_buyers, os['timemode'], os['interval'], shuffle_times, True)

        ordertype = 'Bid'
        (sched, mode) = getschedmode(time, os['dem'])
        for t in range(n_buyers):
            issuetime = time + issuetimes[t]
            tname = 'B%02d' % t
            orderprice = getorderprice(t, sched, n_buyers, mode, issuetime)
            order = Order(tname, ordertype, orderprice, 1, issuetime, -3.14)
            new_pending.append(order)

        # supply side (sellers)
        issuetimes = getissuetimes(n_sellers, os['timemode'], os['interval'], shuffle_times, True)
        ordertype = 'Ask'
        (sched, mode) = getschedmode(time, os['sup'])
        for t in range(n_sellers):
            issuetime = time + issuetimes[t]
            tname = 'S%02d' % t
            orderprice = getorderprice(t, sched, n_sellers, mode, issuetime)
            order = Order(tname, ordertype, orderprice, 1, issuetime, -3.14)
            new_pending.append(order)
    else:
        # there are pending future orders: issue any whose timestamp is in the past
        new_pending = []
        for order in pending:
            if order.time < time:
                # this order should have been issued by now
                # issue it to the trader
                tname = order.tid
                response = traders[tname].add_order(order, verbose)
                if verbose:
                    print('Customer order: %s %s' % (response, order))
                if response == 'LOB_Cancel':
                    cancellations.append(tname)
                    if verbose:
                        print('Cancellations: %s' % cancellations)
                # and then don't add it to new_pending (i.e., delete it)
            else:
                # this order stays on the pending list
                new_pending.append(order)
    return [new_pending, cancellations]

def init_extremes(pe, traders):
    # use pe to determine number of extremists (and hence number of moderates)
    n_extremists=pe*N*2
    N_P_plus=plus_neg[0]*int(0.5+(n_extremists/sum(plus_neg)))
    # N_P_plus=int(0.5 + n_extremists)
    print("N_P_plus: %d" % N_P_plus)

    #assume symmetric plus/minus
    N_P_minus=plus_neg[1]*int(0.5+(n_extremists/sum(plus_neg)))
    # N_P_minus = 0
    print("N_P_minus: %d" % N_P_minus)

    n_moderate=2*N-(N_P_plus+N_P_minus)

    # create extremists: Max then Min
    keys = list(traders.keys())

    for i in range(N_P_plus):
        traders[keys[i]].opinion = Max_Op-(random.random()*extreme_distance)
        traders[keys[i]].uncertainty = u_e
        traders[keys[i]].start_opinion = "extreme"

    for i in range(N_P_minus):
        index = keys[N_P_plus + i]
        traders[index].opinion = Min_Op+(random.random()*extreme_distance)
        traders[index].uncertainty = u_e
        traders[index].start_opinion = "extreme"

    return n_moderate

# one session in the market
def market_session(sess_id, starttime, endtime, trader_spec, order_schedule, tdump, odump, dump_all, verbose, model_name):

    orders_verbose = False
    lob_verbose = False
    process_verbose = False
    respond_verbose = False
    bookkeep_verbose = False
    populate_verbose = False

    # initialise the exchange
    exchange = Exchange()

    # create a bunch of traders
    traders = {}

    trader_stats = populate_market(trader_spec, traders, True, populate_verbose, model_name)

    # timestep set so that can process all traders in one second
    # NB minimum interarrival time of customer orders may be much less than this!!
    timestep = 1.0 / float(trader_stats['n_buyers'] + trader_stats['n_sellers'])

    duration = float(endtime - starttime)

    last_update = -1.0

    time = starttime

    pending_cust_orders = []

    verbose = False

    if verbose:
        print('\n%s;  ' % sess_id)

    current_time = 0

    while time < endtime:

        # how much time left, as a percentage?
        time_left = (endtime - time) / duration


        # if verbose: print('\n\n%s; t=%08.2f (%4.1f/100) ' % (sess_id, time, time_left*100))

        # if ((endtime - time) < duration / 2) and extremes_half == 1:
        global extremes_half
        if (int(sess_id[5:]) == n_trials // 2) and extremes_half == 1:
            # switch plus_neg
            global plus_neg
            plus_neg = [abs(plus_neg[0]-1), abs(plus_neg[1]-1)]
            print("extremed made")
            init_extremes(pe_max, traders)
            extremes_half = 0

        trade = None


        #    distribute any new customer orders to the traders

        [pending_cust_orders, kills] = customer_orders(time, last_update, traders, trader_stats,
                                                       order_schedule, pending_cust_orders, orders_verbose)

        # if any newly-issued customer orders mean quotes on the LOB need to be cancelled, kill them
        if len(kills) > 0:
            # if verbose : print('Kills: %s' % (kills))
            for kill in kills:
                # if verbose : print('lastquote=%s' % traders[kill].lastquote)
                if traders[kill].lastquote is not None:
                    # if verbose : print('Killing order %s' % (str(traders[kill].lastquote)))
                    exchange.del_order(time, traders[kill].lastquote, verbose)

        # get a limit-order quote (or None) from a randomly chosen trader
        tid = list(traders.keys())[random.randint(0, len(traders) - 1)]

        order = traders[tid].getorder(time, time_left, exchange.publish_lob(time, lob_verbose))

        # if verbose: print('Trader Quote: %s' % (order))

        if order is not None:
            if order.otype == 'Ask' and order.price < traders[tid].orders[0].price:
                sys.exit('Bad ask')
            if order.otype == 'Bid' and order.price > traders[tid].orders[0].price:
                sys.exit('Bad bid')
            # send order to exchange
            traders[tid].n_quotes = 1
            trade = exchange.process_order2(time, order, process_verbose)
            if trade is not None:
                # trade occurred,
                # so the counterparties update order lists and blotters
                traders[trade['party1']].bookkeep(trade, order, bookkeep_verbose, time)
                traders[trade['party2']].bookkeep(trade, order, bookkeep_verbose, time)
                if dump_all:
                    trade_stats(sess_id, traders, tdump, time, exchange.publish_lob(time, lob_verbose))

            # traders respond to whatever happened
            lob = exchange.publish_lob(time, lob_verbose)
            for t in traders:
                # NB respond just updates trader's internal variables
                # doesn't alter the LOB, so processing each trader in
                # sequence (rather than random/shuffle) isn't a problem
                traders[t].respond(time, lob, trade, respond_verbose)

        #           Communicate and update opinions
        if time > current_time:
            opinion_stats(sess_id, traders, odump, time)
            current_time += 1

        if model_name == "BC":
            bounded_confidence_step(mu, delta, time, traders)
        elif model_name == "RA":
            relative_agreement_step(mu, traders)
        elif model_name == "RD":
            relative_disagreement_step_mix(mu, lmda, traders)

        time = time + timestep

    # session has ended

    if dump_all:

        # dump the tape (transactions only -- not dumping cancellations)

        #exchange.tape_dump(sess_id+'_transactions.csv', 'w', 'keep')

        # changed second arg to 'a' as appending to file, not overwriting each time
        # sess_id[4:] because need to remove 'sess'

        if(int(sess_id[4:]) == 1):
            exchange.tape_dump('all_transactions.csv', 'w', 'keep')
        else:
            exchange.tape_dump('all_transactions.csv', 'a', 'keep')

        # record the blotter for each trader
        bdump = open(sess_id+'_blotters.csv', 'w')
        for t in traders:
            bdump.write('%s, %d\n'% (traders[t].tid, len(traders[t].blotter)))
            for b in traders[t].blotter:
                bdump.write('%s, Blotteritem, %s\n' % (traders[t].tid, b))
        bdump.close()


    # write trade_stats for this session (NB end-of-session summary only)
    trade_stats(sess_id, traders, tdump, time, exchange.publish_lob(time, lob_verbose))


# opinion_stats()
# dump CSV statistics on opinion distribution to file for later analysis
def opinion_stats(expid, traders, dumpfile, time):
    trader_types = {}
    n_traders = len(traders)
    for t in traders:
            ttype = traders[t].ttype
            if ttype in trader_types.keys():
                    trader_types[ttype]['opinions'].append(traders[t].opinion)
                    trader_types[ttype]['n'] = trader_types[ttype]['n'] + 1
            else:
                    opinions = [traders[t].opinion]
                    trader_types[ttype] = {'n':1, 'opinions': opinions}


    dumpfile.write('%s, %06d, ' % (expid, time))
    for ttype in sorted(list(trader_types.keys())):
            n = trader_types[ttype]['n']
            os = trader_types[ttype]['opinions']
            dumpfile.write('%s, %d,' % (ttype, n))
            for o in os:
                dumpfile.write('%f, ' % o)
    dumpfile.write('\n');


#############################

# # Below here is where we set up and run a series of experiments


if __name__ == "__main__":

    # set up common parameters for all market sessions
    start_time = 0.0
    end_time = 225.0
    duration = end_time - start_time
    # parameters considering opinion dynamics
    Max_Op = 1
    Min_Op = -1
    model_name = "BC"
    #pe_min = 0.025  #range of global proportion of extremists (pe)
    pe_max = 0.1    # pe_max = 0.3
    #pe_steps = 12
    # u_min = 0.2
    u_min = 0.2
    u_max = 2
    u_steps = 19
    #u_e = 0.1 # extremism uncertainty
    extreme_distance = 0 # how close one has to be to be an "extremist"

    # used for later experiments
    #Min_mod_op = Min_Op + extreme_distance
    #Max_mod_op = Max_Op - extreme_distance
    plus_neg = [1, 1] # [1, 1] for both pos and neg extremes respectively
    # whether or not to start with extremes
    extreme_start = 0
    # extremes half way through
    extremes_half = 0
    # intensity of interactions
    mu = 0.2 # used for all models eg. 0.2 - it's w in the paper - i.e., confidence factor
    delta = 0.15 # used for Bounded Confidence Model eg. 0.1
    #lmda = 1 # used for Relative Disagreement Model eg. 0.1
    generation_count = 0

    # schedule_offsetfn returns time-dependent offset, to be added to schedule prices
    def schedule_offsetfn(t):

        pi2 = math.pi * 2
        c = math.pi * 3000
        wavelength = t / c
        gradient = 100 * t / (c / pi2)
        amplitude = 100 * t / (c / pi2)
        offset = gradient + amplitude * math.sin(wavelength * t)
        return int(round(offset, 0))

    # Here is an example of how to use the offset function
    #
    # range1 = (10, 190, schedule_offsetfn)
    # range2 = (200,300, schedule_offsetfn)

    # Here is an example of how to switch from range1 to range2 and then back to range1,
    # introducing two "market shocks"
    # -- here the timings of the shocks are at 1/3 and 2/3 into the duration of the session.
    #
    # supply_schedule = [ {'from':start_time, 'to':duration/3, 'ranges':[range1], 'stepmode':'fixed'},
    #                     {'from':duration/3, 'to':2*duration/3, 'ranges':[range2], 'stepmode':'fixed'},
    #                     {'from':2*duration/3, 'to':end_time, 'ranges':[range1], 'stepmode':'fixed'}
    #                   ]


    # The code below sets up symmetric supply and demand curves at prices from 50 to 150, P0=100

    # range1 = (95, 95, schedule_offsetfn)
    range1 = (10, 190)
    supply_schedule = [ {'from':start_time, 'to':end_time, 'ranges':[range1], 'stepmode':'fixed'}
                      ]

    # range1 = (105, 105, schedule_offsetfn)
    range1 = (10, 190)
    demand_schedule = [ {'from':start_time, 'to':end_time, 'ranges':[range1], 'stepmode':'fixed'}
                      ]

    order_sched = {'sup': supply_schedule, 'dem': demand_schedule,
                   'interval': 30, 'timemode': 'periodic'}
    # Use 'periodic' if you want the traders' assignments to all arrive simultaneously & periodically
    #               'interval': 30, 'timemode': 'periodic'

    # n_trials is how many trials (i.e. market sessions) to run in total
    n_trials = 5
    # n_recorded is how many trials (i.e. market sessions) to write full data-files for
    n_trials_recorded = 5


    #################################################################################################
    # GP Stuff

    # default global vars
    updated_pmin = None
    updated_pmax = None
    current_trader_dir = None
    updated_strat = None

    def P_join(left, right):

        if(type(left) == str):
            left = float(left)

        if(type(right) == str):
            right = float(right)

        # if left and/or right is a number (i.e., strat value or random constant),
        # then convert it into a pmf, then pass both into the parallel join function

        if(isinstance(left, float) or isinstance(left, int)):

            left = prob_mass_fn(left, None, None, 'null', pmin=updated_pmin, pmax=updated_pmax, dirn=current_trader_dir)

        if(isinstance(right, float) or isinstance(right, int)):

            right = prob_mass_fn(right, None, None, 'null', pmin=updated_pmin, pmax=updated_pmax, dirn=current_trader_dir)

        new_pmf =  prob_mass_fn(0.0, left, right, 'parallel', pmin=updated_pmin, pmax=updated_pmax, dirn=current_trader_dir)
        return new_pmf

    def S_join(left, right):

        if(type(left) == str):
            left = float(left)

        if(type(right) == str):
            right = float(right)

        # if left and/or right is a number (i.e., strat value or random constant),
        # then convert it into a pmf, then pass both into the series join function

        if(isinstance(left, float) or isinstance(left, int)):

            left = prob_mass_fn(left, None, None, 'null', pmin=updated_pmin, pmax=updated_pmax, dirn=current_trader_dir)

        if(isinstance(right, float) or isinstance(right, int)):

            right = prob_mass_fn(right, None, None, 'null', pmin=updated_pmin, pmax=updated_pmax, dirn=current_trader_dir)

        new_pmf = prob_mass_fn(0.5, left, right, 'series', pmin=updated_pmin, pmax=updated_pmax, dirn=current_trader_dir)

        return new_pmf


    def D_join(left, right):

        # if left and/or right is a number (i.e., strat value or random constant),
        # then convert it into a pmf, then pass both into the series join function

        if(type(left) == str):
            left = float(left)

        if(type(right) == str):
            right = float(right)

        if(isinstance(left, float) or isinstance(left, int)):

            left = prob_mass_fn(left, None, None, 'null', pmin=updated_pmin, pmax=updated_pmax, dirn=current_trader_dir)

        if(isinstance(right, float) or isinstance(right, int)):

            right = prob_mass_fn(right, None, None, 'null', pmin=updated_pmin, pmax=updated_pmax, dirn=current_trader_dir)

        new_pmf = prob_mass_fn(0.5, left, right, 'difference', pmin=updated_pmin, pmax=updated_pmax, dirn=current_trader_dir)
        return new_pmf

    # these functions all make the particular pmf have a different weight
    def weight_1(pmf):

        if(type(pmf) == str):
            pmf = float(pmf)

        # if pmf is a number, convert it to a pmf

        if(isinstance(pmf, float) or isinstance(pmf,int)):

            pmf = prob_mass_fn(pmf, None, None, 'null', pmin=updated_pmin, pmax=updated_pmax, dirn=current_trader_dir)

        pmf.assign_weight(1)
        return pmf

    def weight_2(pmf):

        if(type(pmf) == str):
            pmf = float(pmf)

        # if pmf is a number, convert it to a pmf

        if(isinstance(pmf, float) or isinstance(pmf,int)):

            pmf = prob_mass_fn(pmf, None, None, 'null', pmin=updated_pmin, pmax=updated_pmax, dirn=current_trader_dir)

        pmf.assign_weight(2)
        return pmf

    def weight_3(pmf):

        if(type(pmf) == str):
            pmf = float(pmf)

        # if pmf is a number, convert it to a pmf

        if(isinstance(pmf, float) or isinstance(pmf,int)):

            pmf = prob_mass_fn(pmf, None, None, 'null', pmin=updated_pmin, pmax=updated_pmax, dirn=current_trader_dir)

        pmf.assign_weight(3)
        return pmf

    def weight_4(pmf):

        if(type(pmf) == str):
            pmf = float(pmf)

        # if pmf is a number, convert it to a pmf

        if(isinstance(pmf, float) or isinstance(pmf,int)):

            pmf = prob_mass_fn(pmf, None, None, 'null', pmin=updated_pmin, pmax=updated_pmax, dirn=current_trader_dir)

        pmf.assign_weight(4)
        return pmf

    pset = gp.PrimitiveSet("MAIN", 1)
    pset.addPrimitive(P_join, 2)
    pset.addPrimitive(S_join, 2)
    pset.addPrimitive(D_join, 2)
    pset.addPrimitive(P_join, 2)
    pset.addPrimitive(S_join, 2)
    pset.addPrimitive(D_join, 2)
    pset.addPrimitive(weight_1, 1)
    pset.addPrimitive(weight_2, 1)
    pset.addPrimitive(weight_3, 1)
    pset.addPrimitive(weight_4, 1)

    pset.addEphemeralConstant("rand101", lambda: random.uniform(-1,1))

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    pmf_spec = None

    def evaluate_ind(individual):

        # update the pmf_specification passed to OPRZI_GP traders
        # pmf_spec is generated from running the compiled GP expression
        print(individual)
        func = toolbox.compile(expr=individual)
        global pmf_spec
        global generation_count
        pmf_spec = func

        generation_count += 1
        print("Individual {} going in to the market...".format(generation_count % 500))

        zero_profit_count = 0

        #################################################################################################

        #buyers_spec = [('GVWY',10),('SHVR',10),('ZIC',10),('O-PRZI',10)]
        #sellers_spec = [('GVWY',10),('SHVR',10),('ZIC',10),('O-PRZI',10)]

        buyers_spec = [('SHVR',2),('GVWY',2),('O-ZIC',3),('ZIC',3)]
        sellers_spec = [('O-PRZI-GP',1),('SHVR',2),('GVWY',2),('O-ZIC',2),('ZIC',3)]

        traders_spec = {'sellers':sellers_spec, 'buyers':buyers_spec}
        # N = population
        N = sellers_spec[0][1] + buyers_spec[0][1]

        # run a sequence of trials, one session per trial

        verbose = True

        tdump=open('avg_balance.csv','w')
        odump=open("csv_files/opinions.csv",'w')

        trial = 1

        oprzi_gp_profit = 0

        while trial < (n_trials+1):
            trial_id = 'sess%04d' % trial

            print("Individual {} session {}".format(generation_count % 500, trial_id))

            if trial > n_trials_recorded:
                dump_all = False
            else:
                dump_all = True

            market_session(trial_id, start_time, end_time, traders_spec, order_sched, tdump, odump, dump_all, verbose, model_name)
            tdump.flush()
            odump.flush()
            trial = trial + 1

            with open('avg_balance.csv') as csv_file:

                csv_reader = csv.reader(csv_file, delimiter=',')

                for row in csv_reader:
                    # average profit for OPRZI_GP is given as the 12th item in each csv row
                    session_profit = float(row[11][1:])

            oprzi_gp_profit += session_profit

            if(session_profit == 0):
                zero_profit_count += 1

        odump.close()
        tdump.close()
        pmf_spec = None

        average_profit = oprzi_gp_profit/n_trials

        if(generation_count == 500):
            file_name = 'gencount.txt'.format(generation_count)
            ggg=open(file_name,'r')
            num = int(ggg.read(10)) + 1
            num = str(num)
            ggg=open(file_name,'w')
            ggg.write(num)
            ggg.close()


        # to reduce the chance of having a "lucky" trader making it through
        # we are looking for a consistent, successful trader, one that makes profits > 0 at least 60% of the time
        if(zero_profit_count >= 3):
            average_profit = 0

        return average_profit,


    toolbox.register("evaluate", evaluate_ind)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

    pop = toolbox.population(n=500)
    hof = tools.HallOfFame(1)

    stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(len)
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    #mstats = tools.MultiStatistics(fitness=stats_fit,)
    mstats.register("avg", numpy.mean)
    mstats.register("std", numpy.std)
    mstats.register("min", numpy.min)
    mstats.register("max", numpy.max)

    #pop, log = profile.run('algorithms.eaSimple(pop, toolbox, cxpb=0.9, mutpb=0.1, ngen=1, stats=mstats, halloffame=hof, verbose=True)')

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.9, mutpb=0.1, ngen=40, stats=mstats, halloffame=hof, verbose=True)

    # mu is The number of individuals to select for the next generation
    # from textbook, the reproduction rate is (1-p) where p = cxpb + mutpb
    # cxpb = 90% (0.9)
    # mutpb = 1% (0.01)
    # so mutation rate = 0.09 * 500 = 45

    #pop, log = algorithms.eaMuPlusLambda(pop, toolbox, mu=45, lambda_=155, cxpb=0.9, mutpb=0.01, ngen=1, stats=mstats, halloffame=hof, verbose=True)

    #population – A list of individuals.
    #toolbox – A Toolbox that contains the evolution operators.
    #mu – The number of individuals to select for the next generation.
    #lambda_ – The number of children to produce at each generation.
    #cxpb – The probability that an offspring is produced by crossover.
    #mutpb – The probability that an offspring is produced by mutation.
    #ngen – The number of generation.
    #stats – A Statistics object that is updated inplace, optional.
    #halloffame – A HallOfFame object that will contain the best individuals, optional.
    #verbose – Whether or not to log the statistics.

    # print info for best solution found:
    best = hof.items[0]

    tree_dump=open('optimum_oprzi_pmf_seller.txt','w')
    tree_dump.write("-- Best Individual = {}".format(best))
    tree_dump.write("-- Best Fitness = {}".format(best.fitness.values[0]))
    tree_dump.write("\n\n\n")

    final_pop = pop
    for ind in pop:

        tree_dump.write("ind = {}".format(ind))
        tree_dump.write("fitness = {}".format(ind.fitness.values[0]))

    # not going to pick hof[0], going to pick top from last generation

    tree_dump.write("\n\n\n")
    tree_dump.write("log = {}".format(log))
