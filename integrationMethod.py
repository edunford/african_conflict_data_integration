'''
General implementation of the meltt method in python for the following analysis. Python implementation seeks to step around inconsistencies in the MELTT package.
'''

import bisect
import math
import pandas as pd
import numpy as np
from collections import OrderedDict

class integrate:
    '''
    Main method for the meltt integration task across multiple precision levels.
    '''
    def __init__(self,data):
        self.data = data
        self.data_key = pd.DataFrame(dict(dataset = ["acled","ged","gtd"],dataset_code = [1,2,3]))
        self.precision_level = None
        self.p_levels = ["exact","adm2","adm1","country"]
        self.p_level_size = {key:[] for key in self.p_levels}
        self.integration_output = {key:{"selected_matches":[],"all_matches":[]} for key in self.p_levels}
        self.selected_matches = None # Cleaned selected matched output
        self.matches = None # Cleaned "all matches" output
        self.match_key = None # Match key.

        # Relevant fields from the original data for the integration task.
        self.base_fields = ['dataset_code','event_id','date','date','longitude','latitude']
        self.taxonomy_fields = ['civilian_actor', 'ethnic_actor',
                                'religious_actor', 'government_actor', 'violent_actor', 'violence',
                                'civilian_violence', 'fatal_none', 'fatal_low', 'fatal_high',]

        # Parameters for the meltt function
        self.k = len(self.taxonomy_fields);
        self.secondary = [0 for i in range(self.k)]
        self.certainty = [1 for i in range(self.k)]
        self.smartmatch = False
        self.partial = 1
        self.silent = False

    def clear_state(self):
        '''
        Clear the operative state of the integration Method;
        Clearing is necessary to ensure consistency when performing numerous integrations
        across different assumption profiles (i.e. looping through assumptions)
        '''
        self.precision_level = None
        self.p_level_size = {key:[] for key in self.p_levels}
        self.integration_output = {key:{"selected_matches":[],"all_matches":[]} for key in self.p_levels}
        self.selected_matches = None
        self.matches = None
        self.match_key = None

    def check_out(self,prec_level=""):
        """Check out a specific precision code level for integration.

        Parameters
        ----------
        prec_level : str
            Specify the geo-spatial precision level: exact, adm2, adm1, or country

        Returns (internally)
        -------
        DataFrame
            Return data for the specific precision code level (with the date converted to an
            integer).
        """
        def lookup(dat,prec_level=""):
            '''Quick lookup method to subset the data.'''
            key = queries[prec_level]
            return dat.merge(key,on=['dataset','geo_precision'],how="inner")

        def dt_to_int(dt_time):
            '''Convert datetime to integer'''
            return int(10000*dt_time.year + 100*dt_time.month + dt_time.day)

        # Specify the geo-spatial precision levels.
        queries = {}
        queries['adm1'] = pd.DataFrame(dict(dataset = ['acled','gtd','ged','ged'],
                                            geo_precision = [3,4,4,5]))
        queries['adm2'] = pd.DataFrame(dict(dataset = ['acled','gtd','gtd','ged'],
                                            geo_precision = [2,2,3,3]))
        queries['country'] = pd.DataFrame(dict(dataset = ['ged','gtd'],
                                               geo_precision = [6,5]))
        queries['exact'] = pd.DataFrame(dict(dataset = ['acled','ged','ged','gtd'],
                                             geo_precision = [1,1,2,1]))

        # Check out a geo-spatial precision level and reassign the longitude and latitude coordinates to correspond with the level.
        if prec_level in ["adm1","adm2"]:
            tmp = lookup(self.data,prec_level=prec_level)
            tmp.loc[:,"longitude"] = tmp[f"{prec_level}_centroid_lon"]
            tmp.loc[:,"latitude"] = tmp[f"{prec_level}_centroid_lat"]
        elif prec_level == "country": # Make sure country mapping aligns
            tmp = lookup(self.data,prec_level=prec_level)
            clocs = tmp[['country','longitude','latitude']].groupby(['country'],as_index=False).head(1).reset_index(drop=True)
            tmp = tmp.drop(['longitude','latitude'],axis=1).merge(clocs,on=['country'])
        else:
            tmp = lookup(self.data,prec_level=prec_level)

        # To integer Convert date
        tmp.date = [dt_to_int(t) for t in pd.to_datetime(tmp.date)]

        # Keep track of the precision level checked out
        self.precision_level = prec_level

        # Merge on data assignment key and return internally
        self.precision_level_data = tmp.merge(self.data_key,on='dataset')

        # Count the number of observation in the precision level
        self.p_level_size[prec_level] = (self.precision_level_data
                                         .groupby('dataset').size()
                                         .reset_index(name="total_entries")
                                         .assign(precision_level = prec_level))

    def weight_scheme(self,weight_dict=None):
        """Weighting scheme for taxonomy levels (we can control which taxonomy levels matter the most).
        In the function, specify how the taxonomy elements are weighted using a dictionary. Function instantly calculates weights of the other dimensions.

        Parameters
        ----------
        weight_dict : dict
            key: taxonomy field name, level: weight; weights cannot exceed 1.

        Returns
        -------
        list
            list of weights for the meltt algorithm.
        """
        weights = OrderedDict({t: 0 for t in self.taxonomy_fields})
        total_fields = len(self.taxonomy_fields)

        if weight_dict == None: # weight all the dimensions equally.
            self.weights = [1 for i in weights.values()]
        else: # else apply the requested weights
            # Assign the requested weights
            total_w = 0; assigned = 0
            for k,w in weight_dict.items():
                weights[k] = w
                total_w += w; assigned += 1
                if total_w > 1:
                    raise ValueError("Weights exceed 1. Lower weights.")

            # Normalize for the rest of the weights (fill with remainder)
            remainder = (total_fields-assigned)
            if remainder > 0:
                fill_weight = (1-total_w)/remainder
                for k,w in weights.items():
                    if w == 0:
                        weights[k] = fill_weight

            self.weights = [i for i in weights.values()]

    def draw_data_pairing(self):
        """Generator that yields two nested list dataset temporally ordered for integration.

        Returns
        -------
        list
            Temporally orderd nested list for two of the dataset combinations assuming there is sufficient data.
        """
        dataset_combinations = [(1,2),(1,3),(2,3)]
        for grab_these_datasets in dataset_combinations:

            # Grab the data pairing for a specific precision level.
            tmp = self.precision_level_data.loc[self.precision_level_data.dataset_code.isin(grab_these_datasets)]

            if len(tmp.dataset_code.unique()) < 2:
                # Only proceed is there are two unique datasets being considered
                continue

            # Sort all values by date (entries must be time ordered)
            tmp = tmp.sort_values('date')

            # Print off the dataset names (that are being integrated)
            d_names = tmp.dataset.unique().tolist()
            print(f"\tIntegrating {d_names[0]} and {d_names[1]}")

            # Specify the relevant fields (window and taxonomy criteria), not algorithm is written with an
            # end and start date in mind, thus the duplication of the date field.
            base = self.base_fields[:]
            base.extend(self.taxonomy_fields)

            # Condense into a nested list
            d = tmp[base].values.tolist()

            yield d

    def meltt(self,twindow=0,spatwindow=0,weight_specification=None):
        """Run the meltt integration method (Donnay et. al 2018)

        Parameters
        ----------
        twindow: int
            Temporal window
        spatwindow: int
            Spatial window
        weight_specification: dict
            key: taxonomy field name, level: weight; weights cannot exceed 1.
        """
        print(f"\nIntegrating data at the '{self.precision_level}' precision level.",end="\n\n")
        for data_pair in self.draw_data_pairing():
            self.weight_scheme(weight_dict=weight_specification)
            meltt = meltt_method(data_pair) # Create Meltt object
            # run method
            matches, selected_matches = meltt.run(twindow,spatwindow,
                                                 self.smartmatch,self.k,self.secondary,
                                                 self.certainty,self.partial,
                                                 self.weights, True, self.silent)
            # Store output
            self.integration_output[self.precision_level]["selected_matches"].extend(selected_matches)
            self.integration_output[self.precision_level]["all_matches"].extend(matches)
        print(f"\nIntegration complete.")

    def collect_matches(self):
        """Collect and clean all matches flagged by the meltt algorithm.

        Converts the nested list output for each precision level into a single DataFrame
        """
        selected_matches = pd.DataFrame()
        matches = pd.DataFrame()
        print("Collecting matches.")
        for level in self.p_levels:
            out_cols = ['dataset','event_id','bestmatch_data','bestmatch_event','bestmatch_score',
                        'runnerUp1_data','runnerUp1_event','runnerUp1_score','runnerUp2_data',
                        'runnerUp2_event','runnerUp2_score','events_matched']
            selected_matches_tmp = pd.DataFrame(self.integration_output[level]['selected_matches'],
                                                columns=out_cols).assign(precision_level = level)
            matches_tmp = pd.DataFrame(self.integration_output[level]['all_matches'],
                                   columns=['dataset1','event_id1','dataset2','event_id2','match_score']).assign(precision_level = level)

            # Row bind back onto the main frame
            selected_matches = pd.concat([selected_matches,selected_matches_tmp],ignore_index=True)
            matches = pd.concat([matches,matches_tmp],ignore_index=True)
        self.selected_matches = selected_matches
        self.matches = matches

    def atb_matches(self,selected_matches):
        '''Locate all "across the board" matches (events that map across all input datasets).

        Reconciling multiple pathway matches: There are four potential pathways for an "across the board" match. (Number denotes dataset, and letter denotes an entry).

            - Pathway 1: match(1a,2b), match(1a,3c), match(2b,3c) ==> 1a-2b-3c
            - Pathway 2: match(1a,2b), match(1a,3c) ==> 1a-2b-3c
            - Pathway 3: match(1a,2b), match(2b,3c) ==> 1a-2b-3c (2b bridges)
            - Pathway 4: match(1a,3c), match(2b,3c) ==> 1a-2b-3c (3c bridges)

        For pathways 3 and 4, dataset 3 and dataset 2 operate as "bridging events" tying 1 to 2 and 1 to 3, respectively. All four match pathways are possible, as a result more than one "across the board" match can be identified for a set of events for any two dataset pairing. For example, 1a can match to 2b, 3c via pathway 2, but then 1a can match to 2b and 2b to 3d via pathway 4.

        When multiple pathways emerge, a choice needs to be made on which pathway to accept. By only choosing pathway 1 (as the MELTT R package does), the assumption is that the leading dataset information is the determinating factor when determining matches. When strictly held, the assumption prevents matches along pathway 3 & 4. Likewise, when strictly assuming any other pathway arrangement.

        Pathway 1 is the most robust of potential pathways. In this arrangement, each dataset entry independently aligns to every other dataset entry. This is ideal as there maximal agreement, but again when strictly held, prevents the occurrence of bridging events (pathways 3 & 4).

        The following procedure scans all across the board (atb) pairings. When multiple pathways are detected for any atb pairing, the procedure scans for a pathway 1 arrangement. If none exists, it preferences a pathway 2 arrangement. If none exists, it takes the remaining pathway. Note that only a pathway 3 or 4 can exist at this point as the existence of both would imply a pathway 1 arrangement exists.

        Note that all pairings that are dropped are still considered down the line as non-atb pairings.
        '''

        # Articulate each pathway
        def pathway1(tmp):
            p2 = pathway2(tmp)
            link2 = tmp.query('dataset == 2')
            link2.columns = ['bestmatch_data_x','bestmatch_event_x','bestmatch_data_y','bestmatch_event_y']
            pathway = link2.merge(p2,on=['bestmatch_data_x','bestmatch_event_x','bestmatch_data_y','bestmatch_event_y'])
            pathway = pathway[p2.columns]
            pathway.columns = [f"v{i+1}" for i in range(len(pathway.columns))]
            return pathway.drop_duplicates()

        def pathway2(tmp,clean_colnames = False):
            p = tmp.query('dataset == 1')
            a = p.query('bestmatch_data == 2')
            b = p.query('bestmatch_data == 3')
            pathway = p[['dataset','event_id']].merge(a,on=['dataset','event_id']).merge(b,on=['dataset','event_id'])
            if clean_colnames:
                pathway.columns = [f"v{i+1}" for i in range(len(pathway.columns))]
            return pathway.drop_duplicates()

        def pathway3(tmp):
            bridge_link1 = tmp.query('bestmatch_data==2')[['bestmatch_data','bestmatch_event']]
            bridge_link1.columns = ['dataset','event_id']
            bridge_link2 = tmp.query('dataset==2')[['dataset','event_id']]
            bridge_link = pd.concat([bridge_link1,bridge_link2],ignore_index=True).drop_duplicates()

            a = tmp.query("dataset == 1 and bestmatch_data ==2").iloc[:,[2,3,0,1]]
            a.columns = tmp.columns
            b = tmp.query("dataset == 2 and bestmatch_data ==3")

            # Map
            pathway = (bridge_link
                        .merge(a,on=['dataset','event_id'])
                        .merge(b,on=['dataset','event_id'])
                        [['bestmatch_data_x','bestmatch_event_x','dataset','event_id','bestmatch_data_y','bestmatch_event_y']])
            pathway.columns = [f"v{i+1}" for i in range(len(pathway.columns))]
            return pathway.drop_duplicates()

        def pathway4(tmp):
            bridge_link = tmp.query('bestmatch_data==3')[['bestmatch_data','bestmatch_event']].drop_duplicates()

            a = tmp.query("dataset == 1 and bestmatch_data ==3")
            b = tmp.query("dataset == 2 and bestmatch_data ==3")

            # Map
            pathway = (bridge_link
                        .merge(a,on=['bestmatch_data','bestmatch_event'])
                        .merge(b,on=['bestmatch_data','bestmatch_event'])
                        [['dataset_x','event_id_x','dataset_y','event_id_y','bestmatch_data','bestmatch_event']])
            pathway.columns = [f"v{i+1}" for i in range(len(pathway.columns))]
            return pathway.drop_duplicates()

        # Generate pathways
        p1 = pathway1(selected_matches).assign(pathway = 1)
        p2 = pathway2(selected_matches,clean_colnames=True)
        p3 = pathway3(selected_matches)
        p4 = pathway4(selected_matches)

        # Gather the pathways
        atb_match = p1.merge(p2,how="outer",on=p1.columns[:6].values.tolist()).fillna(2)
        atb_match = atb_match.merge(p3,how="outer",on=p3.columns.values.tolist()).fillna(3)
        atb_match = atb_match.merge(p4,how="outer",on=p4.columns.values.tolist()).fillna(4).reset_index(drop=True)

        # Reconcilw multiple pathway matches
        atb_match = atb_match.sort_values('pathway') # Order by pathway

        # if multple, take lowest pathway order (i.e. 1, 2, 3, 4)
        atb_match = atb_match.groupby(['v1','v2']).head(1).reset_index(drop=True)
        atb_match = atb_match.groupby(['v3','v4']).head(1).reset_index(drop=True)
        atb_match = atb_match.groupby(['v5','v6']).head(1).reset_index(drop=True)

        return atb_match.drop("pathway",axis=1)


    def build_match_key(self):
        """Build match key: a long form version of presenting all event match pairs.
        """
        def anti_join(A,B,on):
            '''Perform an Anti-Join (all in one dataset but not the other).'''
            all = A.merge(B, on = on, how = 'outer', indicator = True)
            all = all[all._merge == 'left_only'].drop("_merge",axis=1)
            all = all.loc[:,all.sum(axis=0) > 0]
            all.columns = [c.replace("_x","").replace("_y","") for c in all.columns]
            all = all.reset_index(drop=True)
            return all

        if self.selected_matches is None:
            raise InterruptedError("No selected matches on queue. Run collect_matches() prior to this method.")

        print("Building match key.",end=" ")
        # Extract a subset of the selected matches dataframe.
        tmp = self.selected_matches[['dataset','event_id','bestmatch_data','bestmatch_event']]

        # Find the "across the board" (atb) matches
        atb_m = self.atb_matches(tmp)


        # Find all the partial matches (non-atb-matches)
        tmp.columns = [f"v{i+1}" for i in range(len(tmp.columns))]

        overlap = pd.concat([atb_m.loc[:,['v1','v2']],
                             atb_m.loc[:,['v3','v4']].set_axis(["v1",'v2'],axis=1,inplace=False),
                             atb_m.loc[:,['v5','v6']].set_axis(["v1",'v2'],axis=1,inplace=False)],
                            ignore_index=True)

        # only match pairs that aren't located in the atb
        a = anti_join(tmp,overlap,on=["v1",'v2']); a.columns = ['v3','v4','v1','v2']
        b = anti_join(a,overlap,on=["v1",'v2']); b.columns = ['v1','v2','v3','v4']

        # First for the atb matches
        comb = (atb_m.drop(['v1','v3','v5'],axis=1)
                .reset_index()
                .rename(columns={"index":"match_id"})
                .eval("match_id = match_id + 1"))
        cnames = ["match_id"]
        cnames.extend(self.data_key.dataset.values.tolist())
        comb.columns = cnames
        match_key = pd.melt(comb,id_vars="match_id").set_axis(['match_id','dataset','event_id'],axis=1,inplace=False)

        # Then fold in the partials
        max_ind = comb.match_id.max()
        if pd.isna(max_ind):
            max_ind = 0
        c = b.reset_index().rename(columns={"index":"match_id"}).eval(f"match_id = match_id + {max_ind} + 1")
        stack = [c[['match_id','v1','v2']].set_axis(['match_id','dataset_code','event_id'],axis=1,inplace=False),
                 c[['match_id','v3','v4']].set_axis(['match_id','dataset_code','event_id'],axis=1,inplace=False)]
        stack = pd.concat(stack,axis=0,ignore_index=True).merge(self.data_key,on="dataset_code")[['match_id','dataset','event_id']]

        print("Built.")
        # combine together
        self.match_key = pd.concat([match_key,stack],ignore_index=True).sort_values('match_id').reset_index(drop=True)


    def map_match_key_to_data(self):
        '''Merge the match key onto the original data. '''

        if self.match_key is None:
            raise InterruptedError("No match key located. Run build_match_key() prior to this method.")

        dat_w_matchid = self.data.merge(self.match_key,how="left",on=['dataset','event_id'])
        dat_w_matchid.match_id = dat_w_matchid.match_id.fillna(0)

        if dat_w_matchid.shape[0] != self.data.shape[0]:
            raise ValueError("The shape of the data change when the match key merged onto it. Potential issue with the code.")

        # Re order columns
        c_order = ['dataset','event_id','match_id']
        c_order.extend([c for c in dat_w_matchid.columns if c not in c_order])

        # store
        self.data_w_matchid = dat_w_matchid[c_order]


    def summarize_integration(self,path_out=None,verbose=True):
        """Summarize the state of the integration by tracking event matches.

        Locate:
            - the number of matches that occurred across datasets,
            - the number of matched events
            - the number of unique events

        Parameters
        ---------
        verbose : bool
            Print the table.

        Returns
        -------
            Report results as a table.
        """
        overlap_table = (self.match_key
                         [['match_id','dataset']]
                         .assign(n =1)
                         .pivot_table("n",index='match_id', columns='dataset',fill_value=0)
                         .assign(n = lambda x: x.sum(axis=1)).loc[:,"acled":"n"]
                         .groupby(['n','acled','ged','gtd'])
                         .size()
                         .reset_index(name="n_matches")
                         .assign(n_events = lambda x: x.n_matches*x.n)
                         .drop("n",axis=1)
                         .sort_values(['acled','ged','gtd'],ascending=True))

        # Number of unique events
        unique_table = (self.data_w_matchid
                        .query('match_id == 0')[['dataset']]
                        .assign(n=1).groupby(['dataset']).sum()
                        .reset_index().reset_index()
                        .pivot_table("n",index='index',columns='dataset',fill_value=0)
                        .assign(n_matches = 0)
                        .assign(n_events = lambda x: x.sum(axis=1))
                        .assign(acled = lambda x: np.where(x.acled>0,1,0))
                        .assign(ged = lambda x: np.where(x.ged>0,1,0))
                        .assign(gtd = lambda x: np.where(x.gtd>0,1,0))
                        .reset_index(drop=True))
        unique_table.columns = unique_table.columns.rename('')

        # Combine tables
        self.total_match_table = pd.concat([unique_table,overlap_table],ignore_index=True,sort=False)

        if verbose:
            print("\nSummary of the Integration Task",end="\n\n")
            print(self.total_match_table)


    def summarize_overlap(self,path_out=None,verbose=False):
        """Track the total number of proximate events across the different datasets and precision levels. That is, the number of events that fell in the same spatio temporal bucket for the given precision level.

        The table conceptualizes the population of candidate events in the data integration.

        Parameters
        ----------
        verbose : bool
            Print the table.

        Returns
        -------
            Report results as a table.
        """
        # n overlap across datasets
        n_overlap = (self.matches[['dataset1','dataset2','precision_level']]
         .set_axis(['dataset_code','dataset2','precision_level'],axis=1,inplace=False)
         .merge(self.data_key, on ='dataset_code')
         .drop('dataset_code',axis=1)
         .set_axis(['dataset_code','precision_level','dataset'],axis=1,inplace=False)
         .merge(self.data_key, on ='dataset_code')
         .drop('dataset_code',axis=1)
         .groupby(['precision_level','dataset_x','dataset_y'])
         .size()
         .reset_index(name="n_overlap")
         )

        # total n entries
        level_totals = []
        for level in self.p_level_size.values():
            if type(level) != type([]):
                for c in [('acled','ged'),('acled','gtd'),('ged','gtd')]:
                    n = level[level.dataset.isin(c)].total_entries.sum()
                    level_totals.append([level.precision_level[0],c[0],c[1],n])
        level_totals = pd.DataFrame(level_totals,columns=['precision_level','dataset_x','dataset_y','n_total'])

        # Gather the number of proximate (potential matches) across events.
        self.total_overlap_table = (n_overlap
                                    .merge(level_totals,on=['precision_level','dataset_x','dataset_y'])
                                    .assign(prop_proximate = lambda x: x.n_overlap/x.n_total))

        if verbose:
            print(self.total_overlap_table)

    def batch_export(self,dir_out = None):
        """Export the main integration output.

        Parameters
        ----------
        dir_out : str
            path to the target directory.
        """

        print("Exporting data",end="... ")
        # Export integrated data (i.e. original data with match key)
        f = dir_out + "output01_integrated_conflict_event_day_aggregated_data.csv"
        self.data_w_matchid.to_csv(f,index=False)

        # Export selected matches
        f = dir_out + "output02_selected_matches_from_integration_task.csv"
        self.selected_matches.to_csv(f,index=False)

        # Export total matches
        f = dir_out + "output03_all_matches_from_integration_task.csv"
        self.matches.to_csv(f,index=False)

        # Export table of the total number of matched events.
        f = dir_out + "output04_integration_summary_matched_unique_events.csv"
        self.total_match_table.to_csv(f,index=False)

        # Export table of the total number of proximate event (potential matches)
        f = dir_out + "output05_integration_summary_total_event_overlap.csv"
        self.total_overlap_table.to_csv(f,index=False)

        print("Data exported.")


class meltt_method:
    '''
    Main meltt method employed in the R package implementation.
    '''

    def __init__(self,data):
        self.integration_data = data

    def run(self,twindow, spatwindow, smartmatch, k,secondary, certainty, partial, weight, episodal, silent):
        """
        Main method that implements full iterative pairwise data comparison and disambiguation functionality
        :param dict data: input datasets (located in a single matrix)
        :param list names: names of the input datasets
        :param double twindow: temporal proximity cutoff
        :param double spatwindow: spatial proximity cutoff
        :param boolean smartmatch: sets whether or not most closely matching taxonomy level is found iteratively
        :param int k: number of taxonomy dimensions
        :param list secondary: number of levels for each taxonomy dimension
        :param list certainty: specifies the exact taxonomy level to match on if smartmatch = False
        :param int partial: number of dimensions on which no matches are permitted
        :param list weight: weights of secondary event dimensions considered
        :param boolean episodal: sets whether or not code is run for episodal data
        :param boolean silent: determines whether or not progress is shown
        :return: lists of all potential matches and of the best fitting matches selected
        """
        twindow = float(twindow)
        spatwindow = float(spatwindow)
        if k == 1:
            secondary = [int(secondary)]
            certainty = [int(certainty)]
            weight = [weight]
        else:
            secondary = [int(i) for i in secondary]
            certainty = [int(i) for i in certainty]
        secondary.insert(0, 0)
        matches = self.compare(self.integration_data, twindow, spatwindow,smartmatch, k,
                          secondary, certainty, partial, weight, episodal,silent)
        selected_matches = []
        selected_matches = self.select(matches)
        return matches, selected_matches


    def compare(self,data, twindow, spatwindow, smartmatch, k, secondary, certainty, partial, weight, episodal, silent):
        """
        Method implementing the pairwise comparison for a given spatial and temporal comparison horizon
        :param list data: data as (nested) list
        :param double twindow: temporal proximity cutoff
        :param double spatwindow: spatial proximity cutoff
        :param boolean smartmatch: sets whether or not most closely matching taxonomy level is found iteratively
        :param int k: number of taxonomy dimensions
        :param list secondary: number of levels for each taxonomy dimension
        :param list certainty: specifies the exact taxonomy level to match on if smartmatch = False
        :param int partial: number of dimensions on which no matches are permitted
        :param list weight: weights of secondary event dimensions considered
        :param boolean episodal: sets whether or not code is run for episodal data
        :param boolean silent: determines whether or not progress is shown
        :return: list of all potential matches
        """
        matches = []
        col0 = self.column(data, 0)
        datasetindex = list(set(col0))
        datasetindex.sort()
        index1 = [i for i, j in enumerate(col0) if j == datasetindex[0]]
        index2 = [i for i, j in enumerate(col0) if j == datasetindex[1]]
        tick = math.ceil(len(index1)/5)
        for event1index in index1:
            event2counter = 0
            next_smaller_index = bisect.bisect(index2, event1index) - 1
            if next_smaller_index > -1:
                check = 1
                while check == 1:
                    if next_smaller_index - event2counter > -1:
                        event2index = index2[next_smaller_index - event2counter]
                        t_check = abs(data[event1index][2] - data[event2index][2]) <= twindow and abs(
                            data[event1index][3] - data[event2index][3]) <= twindow
                        if episodal == 1:
                            t_check = data[event1index][2] - data[event2index][2] <= twindow
                        spat_check = self.geo_dist(data[event1index][4], data[event1index][5],
                                              data[event2index][4],data[event2index][5]) <= spatwindow
                        if t_check and spat_check:
                            total_fit = 0
                            matched_criteria = 0
                            ind = 6
                            for criteria in range(0, k):
                                if smartmatch == 1:
                                    abort = 0
                                    fit_counter = 0
                                    ind = ind + secondary[criteria]
                                    while abort == 0 and fit_counter < secondary[criteria + 1]:
                                        if data[event1index][ind+fit_counter] == data[event2index][ind+fit_counter]:
                                            abort = 1
                                            total_fit = total_fit + weight[criteria]*fit_counter/float(
                                                max(1, secondary[criteria+1]-1))
                                            matched_criteria = matched_criteria + 1
                                        else:
                                            fit_counter = fit_counter + 1
                                else:
                                    ind = ind + secondary[criteria]
                                    if data[event1index][ind+certainty[criteria]] == data[event2index][ind+certainty[criteria]]:
                                        total_fit = total_fit + weight[criteria]*certainty[criteria]/float(
                                            max(1, secondary[criteria+1]-1))
                                        matched_criteria = matched_criteria + 1
                            if matched_criteria == k:
                                total_fit = total_fit/float(k)
                                matches.append(
                                    [datasetindex[0], data[event1index][1], datasetindex[1], data[event2index][1],
                                     total_fit])
                            elif partial > 0 and matched_criteria + partial == k:
                                total_fit = (total_fit + partial)/float(k)
                                matches.append(
                                    [datasetindex[0], data[event1index][1], datasetindex[1], data[event2index][1],
                                     total_fit])
                        if not data[event1index][2] - data[event2index][2] <= twindow:
                            check = 0
                    if next_smaller_index - event2counter < 0:
                        check = 0
                    event2counter = event2counter + 1
            event2counter = 0
            next_larger_index = bisect.bisect(index2, event1index)
            if next_larger_index < len(index2):
                check = 1
                while check == 1:
                    if next_larger_index + event2counter < len(index2):
                        event2index = index2[next_larger_index + event2counter]
                        t_check = abs(data[event2index][2] - data[event1index][2]) <= twindow and abs(
                            data[event2index][3] - data[event1index][3]) <= twindow
                        if episodal == 1:
                            t_check = data[event2index][3] - data[event1index][3] <= twindow
                        spat_check = self.geo_dist(data[event1index][4], data[event1index][5], data[event2index][4],
                                              data[event2index][5]) <= spatwindow
                        if t_check and spat_check:
                            total_fit = 0
                            matched_criteria = 0
                            ind = 6
                            for criteria in range(0, k):
                                if smartmatch == 1:
                                    abort = 0
                                    fit_counter = 0
                                    ind = ind + secondary[criteria]
                                    while abort == 0 and fit_counter < secondary[criteria + 1]:
                                        if data[event1index][ind+fit_counter] == data[event2index][ind+fit_counter]:
                                            abort = 1
                                            total_fit = total_fit + weight[criteria]*fit_counter/float(
                                                max(1, secondary[criteria+1]-1))
                                            matched_criteria = matched_criteria + 1
                                        else:
                                            fit_counter = fit_counter + 1
                                else:
                                    ind = ind + secondary[criteria]
                                    if data[event1index][ind+certainty[criteria]] == data[event2index][ind+certainty[criteria]]:
                                        total_fit = total_fit + weight[criteria]*certainty[criteria]/float(
                                            max(1, secondary[criteria+1]-1))
                                        matched_criteria = matched_criteria + 1
                            if matched_criteria == k:
                                total_fit = total_fit/float(k)
                                matches.append(
                                    [datasetindex[0], data[event1index][1], datasetindex[1], data[event2index][1],
                                     total_fit])
                            elif partial > 0 and matched_criteria + partial == k:
                                total_fit = (total_fit + partial)/float(k)
                                matches.append(
                                    [datasetindex[0], data[event1index][1], datasetindex[1], data[event2index][1],
                                     total_fit])
                        if not data[event2index][2] - data[event1index][2] <= twindow:
                            check = 0
                    if next_larger_index + event2counter >= len(index2):
                        check = 0
                    event2counter = event2counter + 1
            if event1index > tick and not silent:
                # print(".",end="")
                tick += tick
        return matches


    def select(self,matches):
        """
        Method to identify best fitting matches among potential matches for event data
        :param list matches: list of all potential matches
        :return: list of best fitting potential matches
        """
        if len(matches) > 0:
            unique_indices = self.unique_rows(
                zip(self.column(matches, 0), self.column(matches, 1), self.column(matches, 2), self.column(matches, 3)), return_index=True)
            unique_match = self.asy_columns(unique_indices, matches)
            unique_incidents = self.unique_rows(zip(self.column(matches, 0), self.column(matches, 1)))
            unique_partners = self.unique_rows(zip(self.column(matches, 2), self.column(matches, 3)))
            unique_incidents_lagged = unique_incidents
            unique_partners_lagged = unique_partners
            next_index = 0
            match_out = []
            global_stop = 0
            while len(unique_incidents) > 0 and len(unique_partners) > 0 and global_stop == 0:
                sub1 = self.asy_columns(
                    [k for k, v in enumerate(self.column(unique_match, 0)) if v == unique_incidents[next_index][0]],
                    unique_match)
                sub1 = self.asy_columns([k for k, v in enumerate(self.column(sub1, 1)) if v == unique_incidents[next_index][1]], sub1)
                sub1 = self.asy_columns(self.argsort(self.column(sub1, 0)), sub1)
                iterator = 0
                abort = 0
                while iterator < len(sub1) and abort == 0:
                    entry = sub1[iterator][0:5]
                    incident = sub1[iterator][0:2]
                    partner = sub1[iterator][2:4]
                    if incident in unique_incidents and partner in unique_partners:
                        next_index = unique_incidents.index(incident)
                        if next_index == len(unique_incidents) - 1:
                            next_index = 0
                        unique_incidents.remove(incident)
                        unique_partners.remove(partner)
                        match_out.append(entry)
                        abort = 1
                    else:
                        sub2 = self.asy_columns([k for k, v in enumerate(self.column(unique_match, 2)) if v == sub1[iterator][2]],
                                           unique_match)
                        sub2 = self.asy_columns([k for k, v in enumerate(self.column(sub2, 3)) if v == sub1[iterator][3]], sub2)
                        sub2 = self.asy_columns(self.argsort(self.column(sub2, 4)), sub2)
                        best_sub2 = sub2[0][:]
                        if sub1[iterator][4] < best_sub2[4]:
                            to_remove = [s for s in match_out if match_out[2:4] == best_sub2[2:4]]
                            if len(to_remove) > 0:
                                match_out.remove(to_remove[0])
                                unique_incidents.append(to_remove[0][0:2])
                            next_index = unique_incidents.index(incident)
                            unique_incidents.remove(incident)
                            match_out.append(entry)
                            abort = 1
                        else:
                            iterator = iterator + 1
                            if iterator == len(sub1):
                                next_index = next_index + 1
                                if next_index == len(unique_incidents):
                                    next_index = 0
                                    if unique_incidents == unique_incidents_lagged and unique_partners == unique_partners_lagged:
                                        global_stop = 1
                                    else:
                                        unique_incidents_lagged = list(unique_incidents)
                                        unique_partners_lagged = list(unique_partners)
            output = [[0 for i in range(12)] for e in range(len(match_out))]
            for result in range(0, len(match_out)):
                sub1 = self.asy_columns([k for k, v in enumerate(self.column(unique_match, 0)) if v == match_out[result][0]],
                                   unique_match)
                sub1 = [row[0:5] for row in sub1]
                sub1 = self.asy_columns([k for k, v in enumerate(self.column(sub1, 1)) if v == match_out[result][1]], sub1)
                sub1 = self.asy_columns(self.argsort(self.column(sub1, 4)), sub1)

                ind = sub1.index(match_out[result])
                sub1_dim = len(sub1)

                if sub1_dim < ind + 3:
                    if sub1_dim < ind + 2:
                        sub1.append([0, 0, 0, 0, 0])
                        sub1.append([0, 0, 0, 0, 0])
                    else:
                        sub1.append([0, 0, 0, 0, 0])
                output[result][:] = match_out[result] + sub1[ind + 1][2:5] + sub1[ind + 2][2:5] + [sub1_dim]
        else:
            output = []
        return output

    def geo_dist(self,lat_position, lon_position, lat_target, lon_target):
        """
        Calculates great circle distance using robust numerical approach
        :param float lat_position: latitude of first point (in degree)
        :param float lon_position: longitude of first point (in degree)
        :param float lat_target: latitude of second point (in degree)
        :param float lon_target: longitude of second point (in degree)
        :return:
        """
        a_val = math.radians(lat_position)
        b_val = math.radians(lat_target)
        l_val = math.radians(lon_position) - math.radians(lon_target)
        d_val = math.sqrt(math.pow(math.cos(b_val) * math.sin(l_val), 2) + math.pow(
            math.cos(a_val) * math.sin(b_val) - math.sin(a_val) * math.cos(b_val) * math.cos(l_val), 2))
        d_val = math.atan2(d_val, (math.sin(a_val) * math.sin(b_val)) + math.cos(a_val) * math.cos(b_val) * math.cos(l_val))
        d_val = math.degrees(d_val)
        return d_val * 111.111


    def unique_rows(self,data, return_index=False):
        """
        Return unique indexes or unique data from nested list; comparison across full depth of nested list
        :param iterator data: data structure to be processed
        :param int return_index: True for list of index. False for unique Data
        :return: list of unique row indices or of unique row values
        """
        unique_val = []
        unique_indices = []
        for k, v in enumerate(data):
            if not list(v) in unique_val:
                unique_val.append(list(v))
                unique_indices.append(k)
        if return_index:
            return unique_indices
        else:
            return unique_val


    def argsort(self,seq):
        """
        Emulates numpy args Sort
        :param list seq: sequence of numbers
        :return: argsort numpy format
        """
        return sorted(range(len(seq)), key=seq.__getitem__)


    def asy_columns(self,nested_list, data):
        """
        Receives a nested list in form of dumpy access matrix
        :param list nested_list: columns to be selected
        :param list data: nested data
        :return: data with selected columns from list
        """
        new_data = []
        for i in nested_list:
            new_data.append(data[i][:])
        return new_data


    def column(self,matrix, i):
        """
        Select column from nested list such that columns are processed like an array
        :param list matrix: nested list [[]]
        :param int i: selected column
        :return:  list
        """
        return [row[i] for row in matrix]
