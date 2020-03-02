## EarlyVotingStation

This is a network analysis introduction with QGIS

Open QGIS and *Add Vector Layer* 4 files(votingStations.geojson, precincts.geojson, population.shp, OSSroad.shp) in the basedata directory. All files are in EPSG4326(WGS84).

1. *Centroids*: create centroid point layer from "population" mesh.

	rename the output layer as populationCentroid

2. *Reproject Layer*:  Reproject all layers to EPSG2450(unit=meter projection for Hamamatsu area in Japan)

	rename the output layer for votingStations as votingStations2450

	rename the output layer for OSSroad as OSSroad2450

	rename the output layer for precinct as precinct2450

	rename the output layer for populationCentroid layer as populationCentroid2450

3. *Join attributes by nearest*: join populationCentroid2450 with precinct2450

	rename the output layer for populationCentroid2450 as populationCentroidWithID2450

	check populationCentroidWithID2450 has attribute(field) "precinctID"

BTW QGIS Processing Toolbox has 3 options of network analysis

    1) Shortest path(point to point) 
    2) Shortest path(point to layer) 
    3) Shortest path(layer to point) 

In this case, what we want to do is "layer(population centroids) to layer(*its* voting station)" calculation.

4. Open Python Console/Open calcDrivingTime.py with Editor and Run it

	This will take about 10 minutes

5. Open drivingTime.csv

	check the result. if you find "inf", it means some part of road network is disconnected.

6. *Join attributes by field*: join drivingTime with populationCentroidWithID2350 by field "meshCode"
	You

	check the table has meshCode, precinctID, population, driving time(cost)

	export the table for further analysis.

#### Note

 You have to make "perfect" road network from OpenStreetMap data. All nodes must be connected in a single network. No "island of road" is allowed.

 In this case, I removed highways (voters will not use toll ways, probably), which made ramp(slip road) disconnected from the whote network.
 There are many disconnected road in OSS data, such as roads inside factory, which must be removed beforehand.

 Quality of OSS data depends on the skill of the local community. Check by yourself, for example, if the crossroad is connected at a single shared node. If not, you have to do additional work. (I spent a whole day to make OSSroad.shp for Hamamatsu)






