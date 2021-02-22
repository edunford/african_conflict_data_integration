'''
AIM:
    Read in conflict event data (ACLED, UCDP-GED, GTD), generate reports on the number of duplicate entries (i.e. events that occur at the same spatio-temporal location but contain multiple event entries), and map on the necessary administrative data.

OUTPUT:
    generates a single complete data frame with all three datasets (containing only events that occurred on the African continent) and administrative centroid data for imprecision rounding.

NOTE:
    all dropped entries are tracked in the class object. See `cd.dropped_events` for complete reports of all dropped entries.

TODO:
    - export the dropped events to a single frame for analysis
'''
from ConflictDataResolution import conflictData

# Generate reports (Set to False if duplication reports don't need to be generated).
gen_reports = False

# %% Read in conflict data.
cd = conflictData()
cd.load_conflict_datasets(episode_threshold=7)

# %% Generate duplicate reports: track how many entries contain duplicative entries (i.e. events that occur at the same spatio-temporal time stamp)
if gen_reports:
    cd.report_duplicates(dataset ="acled",criteria = ['actor1',"actor2"],
                         file_out="output_data/duplication_reports/acled_duplicates_version01.txt")
    cd.report_duplicates(dataset ="acled",
                      file_out="output_data/duplication_reports/acled_duplicates_version02.txt")
    cd.report_duplicates(dataset ="gtd",criteria = ['actor1',"actor2"],
                       file_out="output_data/duplication_reports/gtd_duplicates_version01.txt")
    cd.report_duplicates(dataset ="gtd",
                       file_out="output_data/duplication_reports/gtd_duplicates_version02.txt")
    cd.report_duplicates(dataset ="ged",criteria = ['actor1',"actor2"],
                       file_out="output_data/duplication_reports/ged_duplicates_version01.txt")
    cd.report_duplicates(dataset ="ged",
                      file_out="output_data/duplication_reports/ged_duplicates_version02.txt")

# %% Restrict data to African continent (and 1997 onward) and add adminstrative unit data centroids locations.
cd.add_admin_data()

# %% Build out actor categories
cd.generate_actor_dummies()

# %% Aggregate data to the location-day level
cd.aggregate_data()

# %% Export census of dropped observations
cd.export_dropped_observations('output_data/dropped_entries_census.csv')

# %% Export dataset (to aggregate to the location day)
cd.export_data("output_data/conflict_data_location_day_aggregate.csv")
