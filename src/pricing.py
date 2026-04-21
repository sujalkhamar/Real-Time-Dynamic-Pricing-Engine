def calculate_demand(row):
    demand = 0

    if row['lead_time'] < 20:
        demand += 0.3
    if row['total_guests'] >= 3:
        demand += 0.2
    if row['stay_length'] > 4:
        demand += 0.2

    return demand


def cancellation_risk(row):
    if row['lead_time'] > 100:
        return 0.4
    else:
        return 0.1


def dynamic_price(base_price, demand_score, cancellation_risk, season_factor):

    demand_impact = base_price * demand_score
    cancellation_impact = base_price * cancellation_risk * 0.3
    seasonal_impact = base_price * season_factor

    new_price = base_price + demand_impact + seasonal_impact - cancellation_impact

    return round(new_price, 2)