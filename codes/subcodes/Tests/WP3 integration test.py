import math
import random

def WP3_parameters_simplified(choices3: bool=True):
    def Mode_choice(choices3: bool,
                    IS_Gaso: bool, IS_EV: bool, IS_PT: bool, IS_Bike: bool,
                    EV_OWNERSHIP: bool, HAVING_KIDS: bool,
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

    def Plug_in_choices(LOCATION_WORK: bool, LOCATION_SHOPPING: bool, EV_OWNERSHIP: bool, 
                        PARK_TIME: float, COST_SAVING: float,
                        INCOME: float,
                        WALK_TIME: float=0,
                        CYCLE: int=1, SOC: float=50, BATTERY_GUARANTEE: float=50, 
                        DISTANCE_NEXT: float=0):
        
        ASC_PLUGIN          = 0.845600726
        B_LOCATION_WORK     = -0.214813432
        B_LOCATION_SHOPPING = -0.349427058
        B_PARK_TIME         = 0.072730788
        B_DISTANCE_NEXT     = 0.005024964
        B_COST_SAVING       = 0.129340844
        B_WALK_TIME         = -0.082726453
        B_INCOME_LOW        = 0.182566091
        B_INCOME_HIGH       = 0.265787361
        B_CYCLE             = -0.052004313
        B_SOC_EVowner       = -0.004559245
        B_SOC_nonEVowner    = -0.009348043
        B_BATTERY_GUARANTEE_nonEVowner  = 0.004927057

        V_plugin  = (
            ASC_PLUGIN+
            B_LOCATION_WORK * LOCATION_WORK + B_LOCATION_SHOPPING * LOCATION_SHOPPING +
            B_PARK_TIME * PARK_TIME +
            B_DISTANCE_NEXT * DISTANCE_NEXT +
            B_COST_SAVING* COST_SAVING +
            B_WALK_TIME * WALK_TIME +
            B_INCOME_LOW * (INCOME<=3000)+ 
            B_INCOME_HIGH * (INCOME>6000) + 
            B_CYCLE *CYCLE +
            B_SOC_EVowner * (EV_OWNERSHIP==1) * SOC +
            B_SOC_nonEVowner * (EV_OWNERSHIP==0) * SOC + B_BATTERY_GUARANTEE_nonEVowner * (EV_OWNERSHIP==0) * BATTERY_GUARANTEE 
        )
        V_notplugin = 0

        P_plugin = math.exp(V_plugin)/(math.exp(V_plugin) + math.exp(V_notplugin))
        P_notplugin = 1-P_plugin

        #print(f"Probability of plug-in: {P_plugin}")
        #print(f"Probability of not plug-in: {P_notplugin}")

        modes = [True, False]
        plugin = random.choices(modes, weights=[P_plugin, P_notplugin], k=1)[0]
        
        return plugin

    def data_gathering():
        
        data = {}
        
        data['COST']                = 10
        data['IS_Gaso']             = False
        data['IS_EV']               = False
        data['TRAVEL_TIME']         = 20
        data['IS_PT']               = True
        data['IS_Bike']             = False
        data['WALK_TIME']           = 10
        data['COST_EV2G']                   = 2
        data['TRAVEL_TIME_EV2G']            = 10
        data['WALK_TIME_EV2G']              = 5
        data['EV_OWNERSHIP']                = True
        data['HAVING_KIDS']                 = False
        data['COST_CSV2G']              = 3
        data['TRAVEL_TIME_CSV2G']       = 10
        data['WALKING_TIME_CV2G']       = 5
        data['AGE']                     = 30
        data['INCOME']                  = 3500
        data['LOCATION_WORK']       = True
        data['LOCATION_SHOPPING']   = False
        data['PARK_TIME']           = 8

        return data
    
    data = data_gathering()

    chosen_mode = Mode_choice(choices3,
                              data['IS_Gaso'], data['IS_EV'], data['IS_PT'], data['IS_Bike'],
                              data['EV_OWNERSHIP'], data['HAVING_KIDS'],
                              data['AGE'], data['INCOME'], 
                              data['COST'], data['COST_EV2G'], data['COST_CSV2G'], 
                              data['WALK_TIME'], data['WALK_TIME_EV2G'], data['WALKING_TIME_CV2G'], 
                              data['TRAVEL_TIME'], data['TRAVEL_TIME_EV2G'], data['TRAVEL_TIME_CSV2G'])

    #print(f"chosen_mode: {chosen_mode}")

    if (chosen_mode == 'P_1') & (not data['IS_EV']):
        return chosen_mode, False

    if chosen_mode == 'P_2':
        COST_P = data['COST_EV2G']
    else:
        COST_P = data['COST_CSV2G']

    plugin = Plug_in_choices(data['LOCATION_WORK'], data['LOCATION_SHOPPING'], data['EV_OWNERSHIP'], 
                             data['PARK_TIME'], COST_P,
                             data['INCOME'])

    #print(f"plugin: {plugin}")

    return chosen_mode, plugin 

results = WP3_parameters_simplified()

print(results)