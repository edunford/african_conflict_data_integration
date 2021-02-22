# African Conflict Data Integration

The following repository contains code for integrating data for all of Africa as per outlined in our working paper, "An Integrated Picture of Conflict".

Pipeline requires the following inputs:

- raw_data/ folder containing:
	+ a .sqlite db of all conflict data to be integrated. Note the table names will need to be adapted in the source code if table naming conventions are changed.
	+ ADM1 shape files for all adminstration units in Africa (See https://www.diva-gis.org/)
	+ Prio-grid polygons for all grid units corresponding with the relevant geographic unit polygons (see https://grid.prio.org/)
	+ Taxonomy assumptions (Note: these are assumptions, use only if there is agreement)
		+ event taxonomy [provided]
		+ actor taxonomy [provided]

Note that the class method ingests pointers to each required data resource upon instantiation (so adjusting the pointers is straightforward).

# Citation

Dunford, Eric, Karsten Donnay, David Backer, and David E. Cunningham. 2020. "An Integrated Picture of Conflict". Working Paper. 

# Issues/Questions

If you stumble across any issues/bugs, please report them as issues. 

