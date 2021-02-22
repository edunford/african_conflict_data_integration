'''
ConflictResolution is a class for manipulating and evaluating the conflict event data. The resolution is getting the data to work together given all the idiosyncracies, event duplication, and spatio-temporal imprecision.

FEATURES:
    - Read in ACLED, GTD, UCDP-GED
    - Evaluate the scope of duplicate entries (internally) in the data
    - Convert unit of analysis to location day (generalizing event and actor information)
    - Generate a common key of centroid locations for geo-spatially
    imprecise entries to be aggregated to and map conflict events to the common key.
    - Aggregated data down to the location day.
'''
import pandas as pd
import geopandas as gpd
import sqlite3
import os
import re
import numpy as np
from collections import OrderedDict

class conflictData:
    '''
    Streamline the importation and curation of the conflict datasets
    for use in the African integration.
    '''

    def __init__(self):
        self.conn = sqlite3.connect(os.path.expanduser("raw_data/conflict_data.sqlite"))
        self.conn.text_factory = lambda x: str(x, 'latin1')
        self.africa_spatial_data_loc = "raw_data/Africa/africa_complete.shp" # African continent shape file (w/adm1)
        self.africa_centroid_data_loc = "output_data/africa-centroid-database.csv" # Adm1/Adm2 centroid data for Africa.
        self.raw_actor_dict_loc = "raw_data/actor-dictionary-raw.txt" # Raw actor dictionary (adjust this to include new categorizations)
        self.event_taxonomy_loc = "raw_data/event_taxonomy.csv"
        self.dropped_events = {k:[] for k in ["ged","acled","gtd"]}

    def expand_date(self,dat):
        """Unpack the episodic event entries into single day event entries

        Parameters
        ----------
        dat : DataFrame
            Data Frame object with episodic entries
        """
        if "startdate" not in dat.columns:
            return dat

        # Expand data
        data_entries = []
        event_entries = []
        date_entries = []
        for d_ind,e_ind,d1,d2 in zip(dat.dataset,dat.event_id,
                                     pd.to_datetime(dat.startdate),
                                     pd.to_datetime(dat.enddate)):
            if (d2-d1).days <= self.episode_threshold:
                dates = pd.date_range(d1,d2)
                data_entries.extend([d_ind]*len(dates))
                event_entries.extend([e_ind]*len(dates))
                date_entries.extend(dates)
        D = pd.DataFrame(dict(dataset=data_entries,event_id=event_entries,date=date_entries))
        dat2 = dat.drop(columns=['startdate', 'enddate'])
        D = D.merge(dat2, on=['dataset', 'event_id'])

        # Save all dropped events (for record)
        key = set(D.event_id.drop_duplicates().values.tolist())
        drop = list(set(dat2.event_id.drop_duplicates().values.tolist()).difference(key))
        reason = f"Reason Dropped: Episodes that fall outside the {self.episode_threshold} day epsidoe threshold window."
        self.dropped_events[D.dataset[0]].append((reason,dat.query("event_id in @drop")))
        return D

    def query_dataset(self,query):
        '''
        Streamline the reading in of the conflict event dataset.
        Add a unique idenfitying key as data is being loaded in.
        '''
        dat = (pd.read_sql_query(query,self.conn)
         .reset_index()
         .rename(columns={'index':'event_id'})
         .eval('event_id = event_id + 1')
         )
        return self.expand_date(dat)


    def load_conflict_datasets(self,episode_threshold=7):
        """Batch read in the relevant fields from the relevant conflict datasets

        Parameters
        ----------
        episode_threshold : int
            Time threshold (in days) of episodic durations to be considered.
            Past this point episode entires are ignored.
        """
        self.episode_threshold = episode_threshold
        self.conflict_datasets = dict()
        print("Reading in the conflict datasets.")

        print("\tRead in ACLED")
        self.conflict_datasets['acled'] = self.query_dataset(f"""
                        select 'acled' as dataset,
                        event_date as date,
                        case
                            when (event_type='Riots/Protests' AND fatalities = 0) then 'Protest'
                            when (event_type='Riots/Protests' AND fatalities > 0) then 'Riot'
                            when (event_type='Riots/protests' AND fatalities = 0) then 'Protest'
                            when (event_type='Riots/protests' AND fatalities > 0) then 'Riot' else event_type end event_tax,
                        actor1,
                        actor2,
                        geo_precision,
                        fatalities,
                        notes as descrip,
                        longitude,
                        latitude
                        from acled_all_africa_1997_2018;
                      """)

        print("\tRead in GTD")
        self.conflict_datasets['gtd'] = self.query_dataset(f"""
                       select
                       'gtd' as dataset,
                       event_date as date,
                       attacktype1_txt as event_tax,
                       gname as actor1,
                       targtype1_txt  as actor2,
                       specificity as geo_precision,
                       nkill as fatalities,
                       summary as descrip,
                       longitude,
                       latitude
                       from 'gtd-1970-2018'
                       where event_date is not NULL and iyear >= 1990;
                       """)

        print("\tRead in GED")
        self.conflict_datasets['ged'] = self.query_dataset(f"""
                       select
                       'ged' as dataset,
                       date_start as startdate,
                       date_end as enddate,
                       type_of_violence as event_tax,
                       side_a as actor1,
                       side_b  as actor2,
                       where_prec as geo_precision,
                       best as fatalities,
                       source_headline as descrip,
                       longitude,
                       latitude
                       from ucdp_ged_1989_2018;
                       """)
        print("Complete.")

    def stack_conflict_datasets(self):
        """Stack the 3 standardized conflict event datasets.
        """
        print("Stacking conflict datasets.")
        D = pd.DataFrame()
        for key in self.conflict_datasets.keys():
            D = pd.concat([D,self.conflict_datasets[key]],sort=False,ignore_index=False)

        # Flag Episodes
        episodes = D.query('dataset == "ged"').groupby(['dataset','event_id']).size()
        episodes = episodes[episodes>=2]
        D.loc[:,'episode'] = 0
        D.loc[D.set_index(['dataset','event_id']).index.isin(episodes.index),'episode'] = 1

        # Save stacked version of the data
        self.stacked_data = D


    def add_admin_data(self):
        """Maps the conflict event data to the African continent (dropping all entries that do not map), then maps on the administration unit centroid locations each event.

        The administration centroids are required to systematically deal with imprecision in the data, as all imprecise point locations can be aggregated to the same location, despite arbitrary differences in decision rules across datasets.
        """
        self.stack_conflict_datasets()
        data_entries = self.stacked_data

        # Remove all entries without spatial coordinates
        reason = "Reason Dropped: Missing geospatial information. Can't use location data to identify features."
        self.dropped_events['gtd'].append((reason,data_entries[data_entries.longitude.isna()]))
        data_entries = data_entries[~data_entries.longitude.isna()]

        print("Mapping events onto spatial features.",end=" ")
        # Turn into events into spatial features.
        data_entries = gpd.GeoDataFrame(data_entries,
                                         geometry=gpd.points_from_xy(data_entries.longitude,
                                                                     data_entries.latitude))
        data_entries.crs = {'init' :'epsg:4326'}

        # Load africa map
        afr = gpd.read_file(self.africa_spatial_data_loc)

        # keep only events that load to Africa.
        spatial_dat = gpd.sjoin(data_entries,afr, how='inner', op='intersects')

        # Store missing.
        missing = self.anti_join(data_entries, spatial_dat[['dataset','event_id']])
        reason = "Reason Dropped: Event fell outside the African spatial boundary."
        for i in missing.dataset.unique():
            self.dropped_events[i].append((reason,missing.query(f"dataset=='{i}'")))

        # Drop geometry and add centroids
        cent = pd.read_csv(self.africa_centroid_data_loc)
        spatial_dat_nogeo = spatial_dat.drop(['geometry','index_right'],axis=1)
        dat_complete = spatial_dat_nogeo.merge(cent,on=['country','adm1','adm2'],how="left")

        # Check merge occurred correctly.
        if dat_complete.shape[0] != spatial_dat_nogeo.shape[0]:
            raise ValueError("Issue with spatial merge. Observations lost/added.")

        # Drop entries that fall outside the temporal overlap window
        years = pd.to_datetime(dat_complete.date).dt.year
        drop_due_to_time = dat_complete[(years < 1997) | (years > 2018)]
        dat_complete = dat_complete[(years >= 1997) & (years <= 2018)]

        reason = "Reason Dropped: Event does not fall in the 1997 - 2018 temporal window (when all three datasets temporally overlap)."
        for i in drop_due_to_time.dataset.unique():
            self.dropped_events[i].append((reason,drop_due_to_time.query(f"dataset=='{i}'")))

        print("Complete.")
        # Store completed data
        self.complete_data = dat_complete


    def report_duplicates(self,dataset = None,criteria = None,file_out=""):
        """Take a census of duplicative entries in ACLED.
        These are events that contain multiple events as recorded by the location meta-data (e.g. date, longitude, and latitude.)

        Parameters
        ----------
        dataset : str
            name of the input dataset.
        criteria : list
            list containing string of variable parameters to determine duplicate entries on (date, lon, and lat are always included).
        file_out : str
            file path for the report to be exported to.

        Returns
        -------
        file : .txt
            The following function generates a report of as a .txt file regarding when and where these event instances occur in the data.
        """
        def check_reduction(vars):
            '''Gather a percent summary of the duplications given the total.'''
            total = a.shape[0]
            possible_reduction = a[vars].drop_duplicates().shape[0]
            return (round((total-possible_reduction)/total,3),total-possible_reduction)

        # collect the acled data.
        a = self.conflict_datasets[dataset]
        _criteria = ['date','latitude','longitude']
        if criteria is not None:
            _criteria.extend(criteria)

        # locate duplicates
        tmp = a.groupby(_criteria).size().reset_index().rename(columns={0:"n_dups"})
        dups = tmp[tmp['n_dups']>= 2]
        dups = dups.assign(dup_id = [i+1 for i in range(dups.shape[0])])
        a_dups = a.merge(dups,on=_criteria,how="inner").sort_values(['date'])

        # Generate a report of the at issue entries.
        dup_ids = a_dups.dup_id.drop_duplicates().values.tolist()
        indent = "\t\t"
        with open(file_out,"w") as file:
            message = f"""Following tracks all potential duplicative entries in the {dataset.upper()} data. The record reflex that for a subset of events where multiple entries were reported for a single occurrence (an occurrence is defined as having the same spatio-temporal timestamp).

            The following record reflects duplication on the following fields:

                        {_criteria}

            This reflects {check_reduction(_criteria)[0]*100}% ({check_reduction(_criteria)[1]}) of all event entries recorded in {dataset.upper()}.\n\n\n\n"""
            file.writelines(message)
            for id in dup_ids:
                entry = a_dups.query("dup_id == @id")
                file.writelines(f"Date: {entry.date.values[0]} ({entry.longitude.values[0]},{entry.latitude.values[0]}) [geo_prec: {entry.geo_precision.values[0]}]\n\n")
                for i,e in entry.iterrows():
                    file.writelines(f"{indent}[{e.actor1} and {e.actor2}]\n")
                    file.writelines(f"{indent}<<{e.event_tax}>>\n")
                    file.writelines(f"{indent}{indent*2}{e.descrip}\n\n")
                file.writelines(f"--------------------------------\n\n")


    def generate_actor_dummies(self):
        """Build indictors for generalized actor categories.

        Actor categories are specified in the `actor-dictionary-raw.txt` file
        and can be updated there.

        Actor dictionary contains:
            - AMAR (Birnir et. al. 2015) -- for all ethnic categories.
            - All side_b actors in GED when event_type equals 1 or 2 are violent organizations. We parse the names of these organizations and add them to the dictionary of potential violent actors.

        Category flagging logic:
            Iterate through each actor name. See if the name contains a keyword for one of the
            actor category. Actor names can only recieve one designation: civilian, government,
            or violent actor (rebel). However, actors can additionally be coded as religious/ethnic
            if one of those name tags trigger.
        """
        print("Build actor indicators.",end=" ")
        # Open general actor dictionary for rough categorizations of actor names.
        actor_kwds = OrderedDict(); pool = []
        with open(self.raw_actor_dict_loc,"r") as file:
            raw = file.readlines()
            for line in raw:
                if "#" in line:
                    key = line.replace("# ","").strip()
                elif len(line.strip()) > 0:
                    actor_kwds[key] = [i.strip() for i in line.split(',')]
                    pool.extend([i.strip() for i in line.split(',')])

        # Read in complete data
        dat = self.complete_data

        # Replace any missing actor designations with "Unknown"
        if sum(dat.actor1.isna()) > 0:
            dat.actor1.loc[dat.actor1.isna()] = "Unknown"
        if sum(dat.actor2.isna()) > 0:
            dat = dat.assign(actor2 = np.where(dat.actor2.isna(),"Unknown",dat.actor2))

        # Parse actor names with simplified cleaning steps
        actor_names1 = [re.sub(r'[^\w\s]','',re.sub("[\(\[].*?[\)\]]", "", a)).strip().lower() for a in  dat.actor1.unique().tolist()]
        actor_names2 = [re.sub(r'[^\w\s]','',re.sub("[\(\[].*?[\)\]]", "", a)).strip().lower() for a in  dat.actor2.unique().tolist()]

        # Generate a key (to map cleaned actor designations back onto the original data)
        actor1_key = [[i[1],i[0]] for i in zip(actor_names1,dat.actor1.unique())]
        actor2_key = [[i[1],i[0]] for i in zip(actor_names2,dat.actor2.unique())]

        # subset for only the unique actors (store as dictionary)
        actor_names = {actor:[] for actor in set(actor_names1).union(set(actor_names2))}

        # Scan if any of the actor names contain a category flag.
        for actor in actor_names.keys():
            flags = set()
            if actor != "unknown":
                for key, val in actor_kwds.items():
                    for v in val:
                        # if (" " + v + " ") in actor or (" " + v) in actor or (v + " ") in actor:
                        if any([i for i in actor.split() if i == v]):
                            flags.add(key)
            actor_names[actor].extend(flags)

        # Convert category fields into a data frame
        output = []
        category_keys = list(actor_kwds.keys()) # Categories
        for actor, cats in actor_names.items():
            entry = [actor]
            if len(cats) > 0: # If a category was flagged, generate profile
                entry.extend([0 for i in range(len(category_keys))])
                for cat in cats:
                    entry[category_keys.index(cat) + 1] = 1
            else: # Else designation is unknown, so activate all fields (i.e. any are plausible.)
                entry.extend([1 for i in range(len(category_keys))])
            # Generate an output nested list
            output.append(entry)

        # Convert to Pandas data frame.
        cnames = ["actor_clean"]; cnames.extend(category_keys)
        dummy_dat = pd.DataFrame(output,columns=cnames)

        # Map data back onto original naming scheme.
        a1 = pd.DataFrame(actor1_key,columns=["actor1",'actor_clean'])
        a2 = pd.DataFrame(actor2_key,columns=["actor2",'actor_clean'])
        a1_mapped = a1.merge(dummy_dat,on="actor_clean").drop("actor_clean",axis=1)
        a2_mapped = a2.merge(dummy_dat,on="actor_clean").drop("actor_clean",axis=1)

        # Map actor designations back onto original data.
        d1 = dat.merge(a1_mapped,on="actor1",how="left")
        d2 = d1.merge(a2_mapped,on="actor2",how="left")

        # Comb through and combine data.
        rel_cols = list(set([col.replace("_x","").replace("_y","") for col in d2.columns if  "_actor" in col]))
        for r in rel_cols:
            d2.loc[:,r] = np.where(d2[r+"_x"] + d2[r+"_y"]>0,1,0)
            d2 = d2.drop([r+"_x",r+"_y"],axis=1)

        # Build a check in the data.
        if d2.shape[0] != dat.shape[0]:
            raise ValueError("Issue with converting actor names to dummy fields. Actor designations not cleanly mapping back onto the data.")

        print("Complete.")
        # Generate new place holder.
        self.complete_data_with_actor_dummies = d2


    def aggregate_data(self):
        """
        Aggregate data to the location-day

        STEP 1: BUILD SHALLOW EVENT TAXONOMY

            - For the event aggregation step (i.e. aggregating all entries in a specific spatio-temporal location to a single event), events must be standardized into a shallow event taxonomy. The concept mirrors the shallow actor taxonomy: dummy features are activated corresponding to particular event types.

        STEP 2: SIMPLIFIY DATA CATEGORIES
        STEP 3: AGGREGATE TO LOCATION-DAY
        STEP 4: GENERATE SEVERITY DESIGNATIONS (SHALLOW TAX)

            Use the fatalities metric to match events. Broadly clump events into three broad fatality bins:
                - "None" == "No Fatalities"
                - "Low"  == 0 > n_fatal <= 10
                - "High" == 10 > n_fatal

        STEP 5: GENERATE NEW EVENT ID
        STEP 6: EXPORT
        """
        # Aux. functions
        def merge_text(txt):
            '''Simplify text by joining when aggregating'''
            return "; ".join([str(d) for d in txt])

        def clean_agg(val):
            '''Simpify shallow taxonomies'''
            a = sum(val)
            return np.where(a > 0,1,0)

        def adjust_fatal(dat):
            '''Count the number of entries in an episode and
            divide the fatality estimates across the total number
            of episodes in the episode period.
            '''
            a = (dat
                 .query('dataset == "ged" and episode==1')
                 .groupby(['dataset','event_id'])
                 .size()
                 .reset_index(name='n_in_episode'))
            b = dat.merge(a,on=['dataset','event_id'],how="left")
            b.n_in_episode = b.n_in_episode.fillna(1).astype(int)
            cols = b.columns.tolist()
            cols.pop()
            cols.insert(12,"n_in_episode")

            # Divide fatalities by the number of episode entries (uniform distribute
            # the count across the threshold window).
            c = b[cols].assign(fatalities = lambda x: x.fatalities/x.n_in_episode)

            # Return original data with adjustments.
            return c

        print("Aggregating data to the location-day level.",end=" ")
        # Read in the current state of the data and generate a GTD violence against civilians flag
        dat = (self.complete_data_with_actor_dummies
               .eval("vac = (dataset == 'gtd' and civilian_actor==1)")
               .assign(vac = lambda x: x.vac.astype(int))
               .assign(date = lambda x: pd.to_datetime(x.date)))

        # Read in and simplify event taxonomy
        etax = (pd.read_csv(self.event_taxonomy_loc)
             [['data.source','base.categories','Level_2_text']]
             .set_axis(['dataset','event_tax','etype'],axis=1,inplace=False)
             .query('dataset != "scad"'))

        # Simplify event categories
        new = ["irrelevant" if "Nonviolent" in t else "civilian_violence"
               if "Civilians" in t else "violence" for t in etax.etype.unique()]
        etax = (etax.merge(pd.DataFrame(dict(etype=etax.etype.unique(),
                                             new_etype = new)))
                .drop('etype',axis=1)
                .rename(columns={"new_etype":"etype"}))

        # Map to data
        dat.event_tax = dat.event_tax.astype(str)
        dat = dat.merge(etax,on=['dataset','event_tax'])

        # Drop irrelevant entries
        drop_entries = dat.query('etype == "irrelevant"')
        reason = "Reason Dropped: Irrelevant event types with respect to the aim of the analysis."
        for i in drop_entries.dataset.unique():
            self.dropped_events[i].append((reason,drop_entries.query(f"dataset=='{i}'")))
        dat = dat.query('etype != "irrelevant"') # Entries to retain

        # Dropping all GED entries with a 7 precision code. These events fall outside
        # of borders (i.e. international waters or the like). We cannot be certain the
        # event occurred within a specific country. Too imprecise
        drop_entries = dat.query("dataset == 'ged' and geo_precision == 7")
        reason = "Reason Dropped: GED event with a precision code of 7. Event cannot be placed within a country border with any certainty."
        for i in drop_entries.dataset.unique():
            self.dropped_events[i].append((reason,drop_entries.query(f"dataset=='{i}'")))
        dat = dat.query("~(dataset == 'ged' and geo_precision == 7)") # Entries to retain

        # Convert etype to dummies (shallow taxonomy).
        dat2 = (pd.concat([dat.drop(['etype'],axis=1),
                   dat.etype.str.get_dummies()],axis=1)
                .assign(civilian_violence = lambda x: np.where(x.vac==1,1,x.civilian_violence))
                .assign(violence = lambda x: np.where((x.vac==1) & (x.dataset == 'gtd'),0,x.violence))
                .drop('vac',axis=1))

        # convert actor names to a single field
        actor = [", ".join(i) for i in zip(dat2.actor1.values.tolist(),dat2.actor2.values.tolist())]
        dat2 = dat2.drop(['actor1','actor2'],axis=1).assign(actors=actor)

        # Clean existing fields for aggregation
        dat2.fatalities = np.where(dat2.fatalities.isna(),0,dat2.fatalities)
        dat2.descrip = np.where(dat2.descrip.isna(),"No Content.",dat2.descrip)
        dat2.event_id = dat2.event_id.astype(str)
        dat2.adm2 = dat2.adm2.fillna("")
        dat2 = adjust_fatal(dat2)

        # Aggregate data down to location-day
        grp = ['dataset','date','longitude','latitude',
               'country','adm1','adm2',
               'adm1_centroid_lat','adm1_centroid_lon',
               'adm2_centroid_lat','adm2_centroid_lon']

        # Only aggregate data to the location data that has a "precise" geo-code
        precision_key = pd.DataFrame(dict(dataset = ['acled','ged','ged','gtd'],
                                        geo_precision = [1,1,2,1]))
        precise_entries = dat2.merge(precision_key,on=['dataset','geo_precision'])
        imprecise_entries = dat2[~dat2.set_index(['dataset','event_id']).index.isin(precise_entries.set_index(['dataset','event_id']).index)]

        # Check to ensure that the swtich was successful.
        if precise_entries.shape[0] + imprecise_entries.shape[0] != dat2.shape[0]:
            raise ValueError("The splitting of imprecise and precisely geocoded data entries results in the dropping of data values.")

        agg_data = (precise_entries
                    .groupby(grp)
                    .agg(orig_event_id = ("event_id",merge_text),
                         geo_precision=('geo_precision',max),
                         episode = ('episode',max),
                         n_in_episode = ('n_in_episode',max),
                         fatalities=('fatalities',sum),
                         descrip=('descrip',merge_text),
                         actors = ('actors',merge_text),
                         civilian_actor=('civilian_actor',clean_agg),
                         ethnic_actor=('ethnic_actor',clean_agg),
                         religious_actor=('religious_actor',clean_agg),
                         government_actor=('government_actor',clean_agg),
                         violent_actor=('violent_actor',clean_agg),
                         violence=('violence',clean_agg),
                         civilian_violence=('civilian_violence',clean_agg))
                    .reset_index())

        # Re-recombine the precise entries with the imprecise entries.

        agg_data2 = (pd.concat([agg_data,
                                imprecise_entries.rename(columns={"event_id":"orig_event_id"}).drop("event_tax",axis=1)],
                               sort=False,ignore_index=True).reset_index(drop=True))

        # Reset event id index
        agg_data2.loc[:,"event_id"] = pd.Series([i+1 for i in agg_data2.index])

        # Reorder column values
        c_order = ["dataset",'event_id']; c_order.extend([i for i in agg_data2.columns if i not in c_order])
        agg_data2 = agg_data2[c_order]

        # Generate Severity Taxonomy
        agg_data2 = agg_data2.assign(fatal_none = np.where(agg_data2.fatalities==0,1,0))
        agg_data2 = agg_data2.assign(fatal_low = np.where((agg_data2.fatalities <= 10) & (agg_data2.fatal_none==0),1,0))
        agg_data2 = agg_data2.assign(fatal_high = np.where((agg_data2.fatalities > 10),1,0))

        print("Data agregated.")

        # Save data internally
        self.aggregated_data = agg_data2


    def export_data(self,file_path):
        '''Export the final aggregated version of the data'''
        print("Exporting aggregated data.")
        self.aggregated_data.to_csv(file_path,index=False)


    def export_dropped_observations(self,file_path):
        """Export data frame containing all dropped observations. The aim of recording this information is to maintain a census of all observations that are dropped from the analysis and why.

        Parameters
        ----------
        file_path : str
            File path to export the data object.

        Returns
        -------
        file (.csv)
            Export a .csv file with information on the dataset, event_id, date, longitude, and latitude.
        """
        out_dat = pd.DataFrame()
        rel_cols = ['dataset','event_id','date','longitude','latitude']
        for key in self.dropped_events.keys():
            for entry in self.dropped_events[key]:
                reason, tmp = entry
                if "startdate" in tmp.columns:
                    tmp.rename(columns={"startdate":'date'},inplace=True)
                d = tmp[rel_cols].reset_index(drop=True)
                d.loc[:,"drop_reason"] = reason
                out_dat = pd.concat([out_dat,d],sort=True)
        out_dat = out_dat.reset_index(drop=True)[['dataset','event_id','date','longitude','latitude',"drop_reason"]]
        out_dat.loc[:,"step_dropped"] = 1

        print("Exporting 'dropped observation' census.")
        # Export
        out_dat.to_csv(file_path,index=False)

    # Helper Methods
    def anti_join(self,A,B):
        '''Perform an Anti-Join (all in one dataset but not the other).'''
        all = A.merge(B, on = ['dataset','event_id'], how = 'outer', indicator = True)
        return all[all._merge == 'left_only'].drop("_merge",axis=1)
