'''
Run the MELTT integraion task by geo-spatial precision level
'''
import pandas as pd
import os
from integrationMethod import integrate

# Load aggregated version of the data.
dat = pd.read_csv("output_data/conflict_data_location_day_aggregate.csv")
it = integrate(dat)

# integrate by geo-spatial precision levels for different spatio temporal windows
time_windows = [i for i in range(8)] # day ranges to explore
space_windows = [0,5,25,50,100,250] # km ranges to explore
len(time_windows)*len(space_windows) # N iterations

# Weight certain features more in the integration task
weight_specification = dict(violence=.2,
                            civilian_violence=.2,
                            government_actor=.1,
                            violent_actor=.1,
                            civilian_actor=.1)

print("Running large scale integration loop for different spatio-temporal window configurations.")
for t in time_windows:

    print(f"Current Temporal Window: {t} day(s)")

    for s in space_windows:

        print(f"\n\n{'---'*12}\nCurrent Temporal Window: {t} day(s)\nCurrent Spatial Window: {s} km(s)\n{'---'*12}",end="\n\n")

        it.check_out("adm1")
        print(f"\nSpatial window: 0km, temporal window: {t} days")
        it.meltt(twindow=t,spatwindow=0,weight_specification=weight_specification)

        it.check_out("adm2")
        print(f"\nSpatial window: 0km, temporal window: {t} days")
        it.meltt(twindow=t,spatwindow=0,weight_specification=weight_specification)

        it.check_out("country")
        print(f"\nSpatial window: 0km, temporal window: {t} days")
        it.meltt(twindow=t,spatwindow=0,weight_specification=weight_specification)

        it.check_out("exact")
        print(f"\nSpatial window: {s}km, temporal window: {t} days")
        it.meltt(twindow=t,spatwindow=s,weight_specification=weight_specification)

        # Process integration output
        it.collect_matches()
        it.build_match_key()
        it.map_match_key_to_data()

        # Sumamrize the integration output
        it.summarize_integration()
        it.summarize_overlap()

        # Check if storage directory exists
        out_directory = f"output_data/integration_output/s{s}_t{t}/"
        if os.path.isdir(out_directory) == False:
            os.mkdir(out_directory)

        # Export the integration output
        it.batch_export(dir_out=out_directory)

        # Clear state for next integration
        it.clear_state()
