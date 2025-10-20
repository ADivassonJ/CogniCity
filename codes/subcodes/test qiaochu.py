import math
import random

def Mode_choice(choices3: bool,
                IS_Gaso: int, IS_EV: int, IS_PT: int, IS_Bike: int,
                EV_OWNERSHIP: int, HAVING_KIDS: int,
                AGE: int, INCOME: int, 
                COST: float, COST_EV2G: float, COST_CSV2G: float, 
                WALK_TIME: float, WALK_TIME_EV2G: float, WALKING_TIME_CV2G: float, 
                TRAVEL_TIME: float, TRAVEL_TIME_EV2G: float, TRAVEL_TIME_CSV2G: float,
                WAIT_TIME: float=20.5, PARK_COST: float=0, SOC_PEV: float=0):
    
    B_Cost_all          = -0.05 if choices3 else -0.03
    B_Parkcost_GasoEV   = -0.03 
    B_TravelTime_PTBike = -0.02 if choices3 else -0.05
    B_WalkTime_all      = -0.04 if choices3 else -0.03
    B_WaitTime_PT       = -0.16 if choices3 else -0.14
    B_SOC_PEV           = 0.01
    ASC_EV2G_PTUser             = -1.27 if choices3 else 0
    ASC_EV2G_GasoEVUser         = -1.76 if choices3 else 0
    ASC_EV2G_BikeUser           = -2.66 if choices3 else 0
    B_TravelTime_V2G_PTBikeUser = -0.05 if choices3 else 0
    B_EV_OWNERSHIP_EV2G         = 0.52 if choices3 else 0
    B_HAVING_KIDS_EV2G          = -0.37 if choices3 else 0
    ASC_CSV2G_PTUser        = -1.27 if choices3 else -1.17 # SURE?
    ASC_CSV2G_GasoEVUser    = -1.76 if choices3 else 0 # SURE?
    ASC_CSV2G_BikeUser      = -2.66 if choices3 else -3.48
    B_EV_OWNERSHIP_CSV2G    = 0.52 if choices3 else 0.64
    B_HAVING_KIDS_CSV2G     = -0.2 if choices3 else 0
    B_AGE_YOUNG_CSV2G       = 0.21
    B_INCOME_LOW_CSV2G      = 0.35 if choices3 else 0.34


    V1 = (
        B_Cost_all * COST +
        B_Parkcost_GasoEV * PARK_COST * IS_Gaso +
        B_Parkcost_GasoEV * PARK_COST * IS_EV +
        B_TravelTime_PTBike * TRAVEL_TIME * IS_PT +
        B_TravelTime_PTBike * TRAVEL_TIME * IS_Bike +
        B_WalkTime_all * WALK_TIME +
        B_WaitTime_PT * WAIT_TIME * IS_PT +
        B_SOC_PEV * SOC_PEV * IS_EV
    )

    if choices3:
        V2 = (
            ASC_EV2G_PTUser * IS_PT +
            ASC_EV2G_GasoEVUser * IS_EV +
            ASC_EV2G_GasoEVUser * IS_Gaso +
            ASC_EV2G_BikeUser * IS_Bike +
            B_Cost_all * COST_EV2G +
            B_TravelTime_V2G_PTBikeUser * TRAVEL_TIME_EV2G * IS_PT +
            B_TravelTime_V2G_PTBikeUser * TRAVEL_TIME_EV2G * IS_Bike +
            B_WalkTime_all * WALK_TIME_EV2G + 
            B_EV_OWNERSHIP_EV2G * EV_OWNERSHIP + 
            B_HAVING_KIDS_EV2G * HAVING_KIDS 
        )
    else:
        V2 = 0

    V3 = (
        ASC_CSV2G_PTUser*IS_PT +
        ASC_CSV2G_GasoEVUser*IS_EV +
        ASC_CSV2G_GasoEVUser*IS_Gaso +
        ASC_CSV2G_BikeUser*IS_Bike +
        B_Cost_all * COST_CSV2G  +
        B_TravelTime_V2G_PTBikeUser * TRAVEL_TIME_CSV2G * IS_PT +
        B_TravelTime_V2G_PTBikeUser * TRAVEL_TIME_CSV2G * IS_Bike +
        B_WalkTime_all * WALKING_TIME_CV2G +
        B_EV_OWNERSHIP_CSV2G * EV_OWNERSHIP +
        (B_HAVING_KIDS_CSV2G * HAVING_KIDS if choices3 else 0) +
        B_AGE_YOUNG_CSV2G * (AGE <= 35) +
        B_INCOME_LOW_CSV2G * (INCOME <= 3000) 
    )

    #print(f"Utility for choosing the current mode: {V1:.3f}")
    #print(f"Utility for choosing private EV with V2G: {V2:.3f}")
    #print(f"Probability of choosing carsharing services with V2G: {V3:.3f}")

    P_1 = math.exp(V1) / (math.exp(V1) + (math.exp(V2) if choices3 else 0) + math.exp(V3))
    P_2 = (math.exp(V2) if choices3 else 0) / (math.exp(V1) + (math.exp(V2) if choices3 else 0) + math.exp(V3))
    P_3 = math.exp(V3) / (math.exp(V1) + (math.exp(V2) if choices3 else 0) + math.exp(V3))

    #print(f"Probability of choosing current mode: {P_1*100:.3f}%")
    #print(f"Probability of choosing private Evs with V2G: {P_2*100:.3f}%")
    #print(f"Probability of choosing carsharing services with V2G: {P_3*100:.3f}%")
    #print(f"Total: {(P_1+P_2+P_3):.3f}")
    
    modes = ["P_1", "P_2", "P_3"]
    chosen_mode = random.choices(modes, weights=[P_1, P_2, P_3], k=1)[0]
    
    return chosen_mode


choices3 = True


COST                = 10
PARK_COST           = 0
IS_Gaso             = 0
IS_EV               = 0
TRAVEL_TIME         = 20
IS_PT               = 1
IS_Bike             = 0
WALK_TIME           = 10
WAIT_TIME           = 10
SOC_PEV             = 20
COST_EV2G                   = 2
TRAVEL_TIME_EV2G            = 10
WALK_TIME_EV2G              = 5
EV_OWNERSHIP                = 1
HAVING_KIDS                 = 0
COST_CSV2G              = 3
TRAVEL_TIME_CSV2G       = 10
WALKING_TIME_CV2G       = 5
AGE                     = 30
INCOME                  = 3500

chosen_mode = Mode_choice(choices3,
                IS_Gaso, IS_EV, IS_PT, IS_Bike,
                EV_OWNERSHIP, HAVING_KIDS,
                AGE, INCOME, 
                COST, COST_EV2G, COST_CSV2G, 
                WALK_TIME, WALK_TIME_EV2G, WALKING_TIME_CV2G, 
                TRAVEL_TIME, TRAVEL_TIME_EV2G, TRAVEL_TIME_CSV2G)

print(chosen_mode)