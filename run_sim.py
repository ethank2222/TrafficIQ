import os
import sys
import math
import traci

CONFIG_PATH_4_WAY = os.path.join("simulations", "Easy_4_Way", "map.sumocfg")
# 4-Way Intersection light
TL_ID = "TCenter"

# 1. Setup SUMO environment variables
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    # IMPORTANT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Need to update env-vars in your computer, please lmk when you do it so I can help - Erick <3
    sys.exit("Please declare environment variable 'SUMO_HOME'")

def run_simulation(controller: str):
    """Main control loop"""
    # Start SUMO as a subprocess
    if not os.path.exists(CONFIG_PATH_4_WAY):
        print(f"Error: Could not find config file at {CONFIG_PATH_4_WAY}")
        return

    # 'sumo-gui' to see visuals + text
    # 'sumo'     to see text-only
    sumo_config = 'sumo-gui'
    match controller:
        case 'baseline':
            baseline(sumo_config)
        case 'webster':
            webster(sumo_config)
        case _:
            print("Not a valid controller")
    sys.stdout.flush()


def baseline(sumo_config='sumo-gui'):
    traci.start([sumo_config, "-c", CONFIG_PATH_4_WAY])
    step = 0
    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        
        # Logic: Every 200 steps, toggle light
        if step % 200 == 0:
            current_phase = traci.trafficlight.getPhase(TL_ID)
            # Switch between phase 0 (North-South Green) and 2 (East-West Green)
            # 1 & 4 are respective Yellows
            new_phase = 2 if current_phase == 0 else 0
            traci.trafficlight.setPhase(TL_ID, new_phase)
            print(f"Step {step}: Switching traffic light to phase {new_phase}")

        step += 1
    traci.close()


def webster(sumo_config='sumo-gui'):
    MAX_STEP = 3600
    traci.start([sumo_config, "-c", CONFIG_PATH_4_WAY])
    step = 0

    edges = set(traci.edge.getIDList())
    tls = traci.trafficlight.getIDList()

    L = 8  # lost time because 2*4
    fixed_flow = 1850

    north_flow = []
    south_flow = []
    east_flow = []
    west_flow = []
    nf_count = []
    sf_count = []
    ef_count = []
    wf_count = []

    nextCycle = []
    for tl in tls:
        north_flow.append(0)
        south_flow.append(0)
        east_flow.append(0)
        west_flow.append(0)
        nf_count.append(set())
        sf_count.append(set())
        ef_count.append(set())
        wf_count.append(set())
        nextCycle.append(42)
    # total_wait
    sumWait = 0
    total_vehs = 0

    while traci.simulation.getMinExpectedNumber() > 0 and step <= MAX_STEP:
        step += 1
        for tl in range(len(tls)):
            GNS = 0
            GEW = 0
            flowLanes = sorted(set(traci.trafficlight.getControlledLanes(tls[tl])))

            if step == nextCycle[tl]:
                east_flow[tl] = len(ef_count[tl]) * MAX_STEP / step
                north_flow[tl] = len(nf_count[tl]) * MAX_STEP / step
                south_flow[tl] = len(sf_count[tl]) * MAX_STEP / step
                west_flow[tl] = len(wf_count[tl]) * MAX_STEP / step

                sat_north = north_flow[tl] / fixed_flow
                sat_south = south_flow[tl] / fixed_flow
                sat_east = east_flow[tl] / fixed_flow
                sat_west = west_flow[tl] / fixed_flow

                maxOfNS = max(sat_north, sat_south, 0.001)
                maxOfEW = max(sat_east, sat_west, 0.001)
                sumOfAllSats = maxOfNS + maxOfEW
                d = min(42, (1.5 * L + 5) / (1 - sumOfAllSats))
                GNS += (maxOfNS / sumOfAllSats) * (d - L)
                GEW += (maxOfEW / sumOfAllSats) * (d - L)
                nextCycle[tl] += math.ceil(d)

                for e in edges:
                    edgeOcc = traci.edge.getLastStepOccupancy(e)
                    if edgeOcc >= 0.9:
                        if e[0] == 'n' or e[0] == 's':
                            if e[0] == 'e' or e[0] == 'w':
                                continue
                            traci.trafficlight.setPhase(tls[tl], 0)
                        else:
                            traci.trafficlight.setPhase(tls[tl], 2)

                traci.trafficlight.setPhase(tls[tl], 2)
                traci.trafficlight.setPhaseDuration(tls[tl], math.ceil(GEW))
                traci.trafficlight.setPhase(tls[tl], 0)
                traci.trafficlight.setPhaseDuration(tls[tl], math.ceil(GNS))

            ef_count[tl] = ef_count[tl].union(traci.lane.getLastStepVehicleIDs(flowLanes[0]))
            nf_count[tl] = nf_count[tl].union(traci.lane.getLastStepVehicleIDs(flowLanes[1]))
            sf_count[tl] = sf_count[tl].union(traci.lane.getLastStepVehicleIDs(flowLanes[2]))
            wf_count[tl] = wf_count[tl].union(traci.lane.getLastStepVehicleIDs(flowLanes[3]))

        for e in edges:
            vehsNow = traci.edge.getLastStepVehicleIDs(e)
            for id in vehsNow:
                sumWait += traci.vehicle.getWaitingTime(id)
                total_vehs += 1

        traci.simulationStep()

    traci.close()

if __name__ == "__main__":
    run_simulation('webster')
    run_simulation('baseline')