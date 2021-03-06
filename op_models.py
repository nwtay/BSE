import random


#############################

# # Opinion dynamics models

# increment n iterations for traders that communicated
def inc_n_iter(traders):
    for trader in traders:
        trader.n_iter += 1

# bounded confidence model
# (w, delta, k) are confidence factor, deviation threshold and time step respectively
def bounded_confidence_step(w, delta, k, traders):
    # get two traders i and j
    i, j = random.sample(list(traders.keys()), 2)

    X_i = traders[i].opinion
    X_j = traders[j].opinion

    # if difference in opinion is within deviation threshold
    if abs(X_i - X_j) <= delta:

        # trader i update
        i_update = w * X_i + (1 - w) * X_j
        traders[i].set_opinion(i_update)

        # trader j update
        j_update = w * X_j + (1 - w) * X_i
        traders[j].set_opinion(j_update)

    inc_n_iter([traders[i], traders[j]])

def relative_agreement_step(weight, traders):
    i, j = random.sample(list(traders.keys()), 2)

    X_i = traders[i].opinion
    u_i = traders[i].uncertainty

    X_j = traders[j].opinion
    u_j = traders[j].uncertainty

    # Calculate overlap
    h_ij = min((X_i + u_i), (X_j + u_j)) - max((X_i - u_i), (X_j - u_j))
    h_ji = min((X_j + u_j), (X_i + u_i)) - max((X_j - u_j), (X_i - u_i))

    # subtract size of non overlapping part 2ui - hij
    # total agreement between 2 agents: hij - (2ui - hij) = 2(hij - ui)
    # RELATIVE AGREEMENT GIVEN BY: 2(hij - ui) / 2ui = (hij / ui) - 1

    # there's a problem with w > 1.0 where u converges to zero

    # update
    if (h_ji > u_j) :
        RA_ji = (h_ji / u_j) - 1
        traders[i].set_opinion( X_i + (weight * RA_ji * (X_j - X_i)) )
        traders[i].set_uncertainty(u_i + (weight * RA_ji * (u_j - u_i)))
    if (h_ij > u_i) :
        RA_ij = (h_ij / u_i) - 1
        traders[j].set_opinion( X_j + (weight * RA_ij * (X_i - X_j)) )
        traders[j].set_uncertainty(u_j + (weight * RA_ij * (u_i - u_j)))

    inc_n_iter([traders[i], traders[j]])

def relative_disagreement_step(weight, prob, traders):
    i, j = random.sample(list(traders.keys()), 2)

    X_i = traders[i].opinion
    u_i = traders[i].uncertainty

    X_j = traders[j].opinion
    u_j = traders[j].uncertainty

    # Calculate overlap
    # g_ij = max((X_i - u_i), (X_j - u_j)) - min((X_i + u_i), (X_j + u_j))
    g_ij = min((X_i + u_i), (X_j + u_j)) - max((X_i - u_i), (X_j - u_j))
    # g_ji = max((X_j - u_j), (X_i - u_i)) - min((X_j + u_j), (X_i + u_i))
    g_ji = min((X_j + u_j), (X_i + u_i)) - max((X_j - u_j), (X_i - u_i))

    # subtract size of non overlapping part 2ui ??? gij
    # total disagreement given by: gij ??? (2ui ??? gij) = 2(gij ??? ui)
    # RELATIVE DISAGREEMENT GIVEN BY: 2(gij ??? ui) / 2ui = (gij / ui) ??? 1

    # update with prob ??
    if random.random() <= prob:
        if (g_ji > u_j) :
            RD_ji = (g_ji / u_j) - 1
            traders[i].set_opinion( X_i - (weight * RD_ji * (X_j - X_i)) )
            traders[i].set_uncertainty(u_i + (weight * RD_ji * (u_j - u_i)))
        if (g_ij > u_i) :
            RD_ij = (g_ij / u_i) - 1
            traders[j].set_opinion( X_j - (weight * RD_ij * (X_i - X_j)) )
            traders[j].set_uncertainty(u_j + (weight * RD_ij * (u_i - u_j)))

    inc_n_iter([traders[i], traders[j]])

def relative_disagreement_step_mix(weight, prob, traders):
    i, j = random.sample(list(traders.keys()), 2)

    X_i = traders[i].opinion
    u_i = traders[i].uncertainty

    X_j = traders[j].opinion
    u_j = traders[j].uncertainty

    # Calculate overlap
    # x_i - u_i ----- x_i ----- x_i + u_i
    # x_j - u_j ----- x_j ----- x_j + u_j
    # g_ij = max((X_i - u_i), (X_j - u_j)) - min((X_i + u_i), (X_j + u_j))
    g_ij = max((X_i - u_i), (X_j - u_j)) - min((X_i + u_i), (X_j + u_j))
    # g_ji = max((X_j - u_j), (X_i - u_i)) - min((X_j + u_j), (X_i + u_i))
    g_ji = max((X_j - u_j), (X_i - u_i)) - min((X_j + u_j), (X_i + u_i))

    # subtract size of non overlapping part 2ui ??? gij
    # total disagreement given by: gij ??? (2ui ??? gij) = 2(gij ??? ui)
    # RELATIVE DISAGREEMENT GIVEN BY: 2(gij ??? ui) / 2ui = (gij / ui) ??? 1


    # Calculate overlap
    # x_i - u_i ----- x_i ----- x_i + u_i
    # x_j - u_j ----- x_j ----- x_j + u_j
    h_ij = min((X_i + u_i), (X_j + u_j)) - max((X_i - u_i), (X_j - u_j))
    h_ji = min((X_j + u_j), (X_i + u_i)) - max((X_j - u_j), (X_i - u_i))

    # subtract size of non overlapping part 2ui - hij
    # total agreement between 2 agents: hij - (2ui - hij) = 2(hij - ui)
    # RELATIVE AGREEMENT GIVEN BY: 2(hij - ui) / 2ui = (hij / ui) - 1


    # update with prob ??
    if random.random() <= prob:
        if (g_ji > u_j):
            RD_ji = 0
            if u_j != 0: RD_ji = (g_ji / u_j) - 1
            traders[i].set_opinion( X_i - (weight * RD_ji * (X_j - X_i)) )
            traders[i].set_uncertainty(u_i + (weight * RD_ji * (u_j - u_i)))
        if (g_ij > u_i):
            RD_ij = 0
            if u_i != 0: RD_ij = (g_ij / u_i) - 1
            traders[j].set_opinion( X_j - (weight * RD_ij * (X_i - X_j)) )
            traders[j].set_uncertainty(u_j + (weight * RD_ij * (u_i - u_j)))
        # update
    if (h_ji > u_j) :
        RA_ji = (h_ji / u_j) - 1
        traders[i].set_opinion( X_i + (weight * RA_ji * (X_j - X_i)) )
        traders[i].set_uncertainty(u_i + (weight * RA_ji * (u_j - u_i)))
    if (h_ij > u_i) :
        RA_ij = (h_ij / u_i) - 1
        traders[j].set_opinion( X_j + (weight * RA_ij * (X_i - X_j)) )
        traders[j].set_uncertainty(u_j + (weight * RA_ij * (u_i - u_j)))

    inc_n_iter([traders[i], traders[j]])
