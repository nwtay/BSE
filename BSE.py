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

import timeit
import statistics
import sys
import math
import random
from traders import Order, Trader, Trader_Giveaway, Trader_Shaver, Trader_Sniper, Trader_ZIC, Trader_ZIP, Trader_opinionated_ZIC, Trader_OPRZI, Trader_PRZI, Trader_OPRZI_GP
from op_models import bounded_confidence_step, relative_agreement_step, relative_disagreement_step, relative_disagreement_step_mix

bse_sys_minprice = 1  # minimum price in the system, in cents/pennies
bse_sys_maxprice = 500  # maximum price in the system, in cents/pennies
ticksize = 1  # minimum change in price, in cents/penniess

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


##########################---Below lies the experiment/test-rig---##################


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
    def trader_type(robottype, name, Min_Op, Max_Op, u_min, u_max, model, start_opinion, extreme_distance, truth_prevalence):
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
                return Trader_OPRZI('O-PRZI', name, 0.00, 0, 3, opinion, uncertainty, Min_Op, Max_Op, start_opinion, truth_prevalence)
        elif robottype == 'O-PRZI-GP':
                return Trader_OPRZI_GP('O-PRZI-GP', name, 0.00, 0, 3, opinion, uncertainty, Min_Op, Max_Op, start_opinion, truth_prevalence)
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

    # Assigning truth_prevalence based on args passed into prog
    # this is the number of truth seekers in a population of buyers or sellers
    # i.e., num_truth_seekers = 50 means 50 buyers are truth seekers and 50 sellers are truth seekers
    num_truth_seekers = 0
    truth_seeking_prev = "random"

    n_buyers = 0
    for bs in traders_spec['buyers']:
        ttype = bs[0]
        for b in range(bs[1]):
            if(b >= num_truth_seekers):
                # when we have run out of truth seekers, make truth prevalence 0
                tname = 'B%02d' % n_buyers  # buyer i.d. string
                traders[tname] = trader_type(ttype, tname, Min_Op, Max_Op, u_min, u_max, model, "moderate", extreme_distance, 0)
                n_buyers = n_buyers + 1
            else:
                # if < num_truth_seekers, this trader we are about to init needs to have truth_prevalence set to var truth_prevalence
                tname = 'B%02d' % n_buyers  # buyer i.d. string
                if(truth_seeking_prev == "random"):
                    truth_prevalence = random.random()
                else:
                    truth_prevalence = float(truth_prevalence)
                traders[tname] = trader_type(ttype, tname, Min_Op, Max_Op, u_min, u_max, model, "moderate", extreme_distance, truth_prevalence)
                n_buyers = n_buyers + 1


    if n_buyers < 1:
        sys.exit('FATAL: no buyers specified\n')

    if shuffle:
        shuffle_traders('B', n_buyers, traders)

    n_sellers = 0
    for ss in traders_spec['sellers']:
        ttype = ss[0]
        for s in range(ss[1]):
            if(s >= num_truth_seekers):
                tname = 'S%02d' % n_sellers  # buyer i.d. string
                traders[tname] = trader_type(ttype, tname, Min_Op, Max_Op, u_min, u_max, model, "moderate", extreme_distance, 0)
                n_sellers = n_sellers + 1
            else:
                # if < num_truth_seekers, this trader we are about to init needs to have truth_prevalence set to var truth_prevalence
                tname = 'S%02d' % n_sellers  # buyer i.d. string
                if(truth_seeking_prev == "random"):
                    truth_prevalence = random.random()
                else:
                    truth_prevalence = float(truth_prevalence)
                traders[tname] = trader_type(ttype, tname, Min_Op, Max_Op, u_min, u_max, model, "moderate", extreme_distance, truth_prevalence)
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

    def getIntelligentExtremists(num_is):

        if(type(num_is) != int or num_is > len(traders)):
            sys.exit("Invalid number of intelligent extremists: must be int < total pop")

        intelligent_Extremists_list = []

        cnt = 0
        for t in traders:
            if(cnt < num_is):
                intelligent_Extremists_list.append(t)
                cnt += 1

        return intelligent_Extremists_list


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

    if verbose:
        print('\n%s;  ' % sess_id)

    current_time = 0

    # flag to see if shock has been activated
    shock_activated = False

    while time < endtime:

        # when implementing shocks, an adjustment of how traders
        # update their local and global opinions will need to be configured
        # for example, when implementing a particular shock, you may want
        # local opinions to only update after the shock, therefore a flag
        # could be set inside each trader, and an if statement could be
        # constructed in a trader's set_opinion fn. Another example: for MLOFI
        # shocks, you will need to reassing the get_offset value to this shock value
        # Below I provide the framework for instantiating shocks, but the implementation of
        # shocks needs to be configured

        # element 2 of shock list is always the time
        if(shock is not None and shock_activated == False):
            if(time > shock[1]):
                # element 3 is always value
                for t in traders:
                    # implementation of shocks has only been constructed for O-PRZI varaints
                    if(traders[t].ttype == 'O-PRZI' or traders[t].ttype == 'O-PRZI-GP'):
                        if(shock[0] == "MLOFI"):
                            # implement MLOFI shock
                            traders[t].execMlofiShock(shock[2])
                        elif(shock[0] == "LocOp"):
                            # implement local opinion shock
                            traders[t].execLocOpShock(shock[2])
                        elif(shock[0] == "intExtr"):
                            influencers_chosen = getIntelligentExtremists(shock[3])
                            if(t in influencers_chosen):
                                # make this trader become an intelligent extremist
                                traders[t].activateIntExtremists(shock[2], shock[1])
                        else:
                            sys.exit('Shock type not recognised')
                shock_activated = True

        # how much time left, as a percentage?
        time_left = (endtime - time) / duration

        #print(time_left)

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
    end_time = 1000.0
    duration = end_time - start_time
    # parameters considering opinion dynamics
    Max_Op = 1
    Min_Op = -1
    model_name = "BC"
    #pe_min = 0.025  #range of global proportion of extremists (pe)
    #pe_max = 0.1    # pe_max = 0.3
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
    mu = 0.4 # used for all models eg. 0.2 - it's w in the paper - i.e., confidence factor
    delta = 0.5 # used for Bounded Confidence Model eg. 0.1
    lmda = 1 # used for Relative Disagreement Model eg. 0.1

    # for shocks implemented in this project
    # shock (a list of 2+ elements) can be one of three types:
    # -> MLOFI - this changes the offset to a specified value
    # -> Local opinion - changes all local opinions to a specified value
    # -> Influencer - introduces a number of intelligent extremists with a given dir at t
    # a shock is a list of elements, where the structure is as follows:
    # shock[0] is shock type; "MLOFI", "LocOp", "intExtr"
    # shock[1] is the time the shock is introduced (< duration)
    # shock[2] is the corresponding value of the shock, for intExtr this is the op value
    #     that intelligent extremists tend towards
    # if we're doing a intExtr shock, then the shock data type has another element, which
    # is n_I - the number of intelligent extremists, < total pop

    # here are 3 examples
    # shock = ["MLOFI", 20, 50]
    # shock = ["LocOp", 20, -0.5]
    shock = ["intExtr", 300, 1, 9]

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
                   'interval': 1, 'timemode': 'periodic'}

    # Use 'periodic' if you want the traders' assignments to all arrive simultaneously & periodically
    #               'interval': 30, 'timemode': 'periodic'}

    #buyers_spec = [('SHVR',2),('GVWY',2),('O-ZIC',3),('ZIC',3)]
    #sellers_spec = [('O-PRZI-GP',1),('SHVR',2),('GVWY',2),('O-ZIC',2),('ZIC',3)]

    buyers_spec = [('O-PRZI-GP',5)]
    sellers_spec = [('O-PRZI-GP',5)]

    traders_spec = {'sellers':sellers_spec, 'buyers':buyers_spec}
    # N = population
    N = sellers_spec[0][1] + buyers_spec[0][1]

    # run a sequence of trials, one session per trial

    verbose = True

    # n_trials is how many trials (i.e. market sessions) to run in total
    n_trials = 250

    # n_recorded is how many trials (i.e. market sessions) to write full data-files for
    n_trials_recorded = 250

    tdump=open('avg_balance.csv','w')
    odump=open("csv_files/opinions.csv",'w')

    trial = 1

    while trial < (n_trials+1):

        trial_id = 'sess%04d' % trial

        if trial > n_trials_recorded:
            dump_all = False
        else:
            dump_all = True

        market_session(trial_id, start_time, end_time, traders_spec, order_sched, tdump, odump, dump_all, verbose, model_name)
        tdump.flush()
        odump.flush()
        trial = trial + 1

    odump.close()
    tdump.close()
