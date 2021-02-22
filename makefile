# Make the integration pipeline (Steps of the Data Generation are Ordered)
run-integration:
	python3 00_generate_centroid_database.py
	python3 01_combine_conflict_datasets.py
	python3 02_integration.py
