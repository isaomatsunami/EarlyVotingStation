#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  calculate distance/duration to the polling stations

import os, time, csv, math
from qgis.core import *
from qgis.gui import *
from qgis.analysis import *
from qgis.PyQt.QtCore import *
from qgis.PyQt.QtGui import *

"""
***************************************************************************
    Qneat3Framework.py
    Qneat3Utilities.py
    
    Date                 : January 2018
    Copyright            : (C) 2018 by Clemens Raffler
    Email                : clemens dot raffler at gmail dot com
***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

def AssignAnalysisCrs(vlayer):
    logPanel("Setting analysis CRS")
    AnalysisCrs = vlayer.crs()
    return AnalysisCrs

def logPanel(message):
    QgsMessageLog.logMessage(message, "QNEAT3")
    
def isGeometryType(vlayer, type_obj):
    geom_type = vlayer.geometryType()
    if geom_type == type_obj:
        return True
    else:
        return False

def buildQgsVectorLayer(string_geomtype, string_layername, crs, feature_list, list_qgsfield):
    
    #create new vector layer from self.crs
    vector_layer = QgsVectorLayer(string_geomtype, string_layername, "memory")
    
    #set crs from class
    vector_layer.setCrs(crs)
    
    #set fields
    provider = vector_layer.dataProvider()
    provider.addAttributes(list_qgsfield) #[QgsField('fid',QVariant.Int),QgsField("origin_point_id", QVariant.Double),QgsField("iso", QVariant.Int)]
    vector_layer.updateFields()
    
    #fill layer with geom and attrs
    vector_layer.startEditing()
    for feat in feature_list:
        vector_layer.addFeature(feat, True)
    vector_layer.commitChanges()

    return vector_layer

def getFeatureFromPointParameter(qgs_point_xy):     
    feature = QgsFeature()
    fields = QgsFields()
    fields.append(QgsField('point_id', QVariant.String, '', 254, 0))
    feature.setFields(fields)
    feature.setGeometry(QgsGeometry.fromPointXY(qgs_point_xy))
    feature['point_id']="Start Point"
    return feature

def getFeaturesFromQgsIterable(qgs_feature_storage):#qgs_feature_storage can be any vectorLayer/QgsProcessingParameterFeatureSource/etc
    fRequest = QgsFeatureRequest().setFilterFids(qgs_feature_storage.allFeatureIds())
    return qgs_feature_storage.getFeatures(fRequest)

def mergeFeaturesFromQgsIterable(qgs_feature_storage_list):
    result_feature_list = []
    for qgs_feature_storage in qgs_feature_storage_list:
        fRequest = QgsFeatureRequest().setFilterFids(qgs_feature_storage.allFeatureIds())
        result_feature_list.extend(qgs_feature_storage.getFeatures(fRequest))
    return result_feature_list

def getFieldIndexFromQgsProcessingFeatureSource(feature_source, field_name):
    if field_name != "":
        return feature_source.fields().lookupField(field_name)
    else:
        return -1
    
def getListOfPoints(qgs_feature_storage): #qgs_feature_storage can be any vectorLayer/QgsProcessingParameterFeatureSource/etc
    given_geom_type = QgsWkbTypes().displayString(qgs_feature_storage.wkbType()) #GetStringRepresentation of WKB Type
    expected_geom_type = QgsWkbTypes.displayString(1) #Point
    
    if given_geom_type == expected_geom_type: 
        qgsfeatureiterator = getFeaturesFromQgsIterable(qgs_feature_storage)
        return [f.geometry().asPoint() for f in qgsfeatureiterator]
    else:
        pass
        # raise Exception("Unknown geometry", given_geom_type, expected_geom_type)
        
def getFieldDatatype(qgs_feature_storage, fieldname):
    fields_list = qgs_feature_storage.fields()
    qvariant_type = fields_list.field(fieldname).type()
    return qvariant_type

def getFieldDatatypeFromPythontype(pythonvar):
    if isinstance(pythonvar, str):
        return QVariant.String
    elif isinstance(pythonvar, int):
        return QVariant.Int
    elif isinstance(pythonvar, float):
        return QVariant.Double
    else: 
        return QVariant.String

class Qneat3Network():
    """
    Qneat3Network:
    Provides basic logic for more advanced network analysis algorithms
    """

    def __init__(self, 
                 input_network, #QgsProcessingParameterFeatureSource
                 input_points, #[QgsPointXY] or QgsProcessingParameterFeatureSource or QgsVectorLayer --> Implement List of QgsFeatures [QgsFeatures]
                 input_strategy, #int
                 input_directionFieldName, #str, empty if field not given
                 input_forwardValue, #str
                 input_backwardValue, #str
                 input_bothValue, #str
                 input_defaultDirection, #int
                 input_analysisCrs, #QgsCoordinateReferenceSystem
                 input_speedField, #str
                 input_defaultSpeed, #float
                 input_tolerance, #float
                 feedback #feedback object from processing (log window)
                 ): 
        
        """
        Constructor for a Qneat3Network object.
        @type input_network: QgsProcessingParameterFeatureSource
        @param input_network: input network dataset from processing algorithm 
        @type input_points: QgsProcessingParameterFeatureSource/QgsVectorLayer/[QgsPointXY]
        @param input_points: input point dataset from processing algorithm
        @type input_strategy: int
        @param input_strategy: Strategy parameter (0 for distance evaluation, 1 time evaluation)
        @type directionFieldName: string
        @param directionFieldName: Field name of field containing direction information
        @type input_forwardValue: string
        @param input_forwardValue: Value assigned to forward-directed edges
        @type input_backwardValue: string
        @param input_backwardValue: Value assigned to backward-directed edges
        @type input_bothValue: string
        @param input_bothValues: Value assigned to undirected edges (accessible from both directions)
        @type input_defaultDirection: QgsVectorLayerDirector.DirectionForward/DirectionBackward/DirectionBoth
        @param input_defaultDirection: QgsVectorLayerDirector Direction enum to determine default direction
        @type input_analysisCrs: QgsCoordinateReferenceSystem
        @param input_analysisCrs: Analysis coordinate system
        @type input_speedField: string
        @param input_speedField: Field name of field containing speed information
        @type input_tolerance: float
        @param input_tolerance: tolerance value when connecting graph edges
        @type feedback: QgsProcessingFeedback
        @param feedback: feedback object from processing algorithm
        """
        
        #initialize feedback
        self.feedback = feedback
        
        self.feedback.pushInfo("[QNEAT3Network][__init__] Setting up parameters")
        self.AnalysisCrs = input_analysisCrs
        
        #enable polygon calculation in geographic coordinate systems
        distUnit = self.AnalysisCrs.mapUnits()
        self.meter_to_unit_factor = QgsUnitTypes.fromUnitToUnitFactor(QgsUnitTypes.DistanceMeters, distUnit)
        
        #init direction fields
        self.feedback.pushInfo("[QNEAT3Network][__init__] Setting up network direction parameters")
        self.directedAnalysis = self.setNetworkDirection((input_directionFieldName, input_forwardValue, input_backwardValue, input_bothValue, input_defaultDirection))
        self.director = QgsVectorLayerDirector(input_network,
                                    getFieldIndexFromQgsProcessingFeatureSource(input_network, input_directionFieldName),
                                    input_forwardValue,
                                    input_backwardValue,
                                    input_bothValue,
                                    input_defaultDirection)

        #init analysis points
        self.feedback.pushInfo("[QNEAT3Network][__init__] Setting up analysis points")
        if isinstance(input_points,(list,)):
            self.list_input_points = input_points #[QgsPointXY]
        else:
            self.list_input_points = getListOfPoints(input_points) #[QgsPointXY]
            self.input_points = input_points
    
        #Setup cost-strategy pattern.
        self.feedback.pushInfo("[QNEAT3Network][__init__] Setting analysis strategy: {}".format(input_strategy))
        self.default_speed = input_defaultSpeed
        
        self.setNetworkStrategy(input_strategy, input_network, input_speedField, input_defaultSpeed)

        #add the strategy to the QgsGraphDirector
        self.director.addStrategy(self.strategy)
        self.builder = QgsGraphBuilder(self.AnalysisCrs)
        #tell the graph-director to make the graph using the builder object and tie the start point geometry to the graph
        
        self.feedback.pushInfo("[QNEAT3Network][__init__] Start tying analysis points to the graph and building it.")
        self.feedback.pushInfo("[QNEAT3Network][__init__] This is a compute intensive task and may take some time depending on network size")
        start_local_time = time.localtime()
        start_time = time.time()
        self.feedback.pushInfo("[QNEAT3Network][__init__] Start Time: {}".format(time.strftime(":%Y-%m-%d %H:%M:%S", start_local_time)))
        self.feedback.pushInfo("[QNEAT3Network][__init__] Building...")
        self.list_tiedPoints = self.director.makeGraph(self.builder, self.list_input_points, self.feedback)
        self.network = self.builder.graph()
        end_local_time = time.localtime()
        end_time = time.time()
        self.feedback.pushInfo("[QNEAT3Network][__init__] End Time: {}".format(time.strftime(":%Y-%m-%d %H:%M:%S", end_local_time)))
        self.feedback.pushInfo("[QNEAT3Network][__init__] Total Build Time: {}".format(end_time-start_time))
        self.feedback.pushInfo("[QNEAT3Network][__init__] Analysis setup complete")
        
            
    def setNetworkDirection(self, directionArgs):    
        if directionArgs.count("") == 0:
            self.directedAnalysis = True
            self.directionFieldId, self.input_forwardValue, self.input_backwardValue, self.input_bothValue, self.input_defaultDirection = directionArgs
        else:
            self.directedAnalysis = False
            
    def setNetworkStrategy(self, input_strategy, input_network, input_speedField, input_defaultSpeed):
        distUnit = self.AnalysisCrs.mapUnits()
        unit_to_meter_factor = QgsUnitTypes.fromUnitToUnitFactor(distUnit, QgsUnitTypes.DistanceMeters)
        
        speedFieldId = getFieldIndexFromQgsProcessingFeatureSource(input_network, input_speedField)
        if input_strategy == 0:
            self.strategy = QgsNetworkDistanceStrategy()
            self.strategy_int = 0
        else:
            self.strategy = QgsNetworkSpeedStrategy(speedFieldId, float(input_defaultSpeed), unit_to_meter_factor * 1000.0 / 3600.0)
            self.strategy_int = 1
        self.multiplier = 3600

    def calcDijkstra(self, startpoint_id, criterion):
        """Calculates Dijkstra on whole network beginning from one startPoint. Returns a list containing a TreeId-Array and Cost-Array that match up with their indices [[tree],[cost]] """
        tree, cost = QgsGraphAnalyzer.dijkstra(self.network, startpoint_id, criterion)
        dijkstra_query = list()
        dijkstra_query.insert(0, tree)
        dijkstra_query.insert(1, cost)
        return dijkstra_query
    
    def calcShortestTree(self, startpoint_id, criterion):
        tree = QgsGraphAnalyzer.shortestTree(self.network, startpoint_id, criterion)
        return tree
        
    def calcIsoPoints(self, analysis_point_list, max_dist):
        iso_pointcloud = dict()

        for counter, point in enumerate(analysis_point_list):
            self.feedback.pushInfo("[QNEAT3Network][calcIsoPoints] Processing Point {}".format(counter))
            dijkstra_query = self.calcDijkstra(point.network_vertex_id, 0)
            tree = dijkstra_query[0]
            cost = dijkstra_query[1]
            
            current_start_point_id = point.point_id #id of the input point
            current_vertex_id = point.network_vertex_id
            entry_cost = point.entry_cost
            
            field_type = getFieldDatatypeFromPythontype(current_start_point_id)
            
            #startpoints are not part of the Query so they have to be added manually before
            #dikstra is called.
            start_vertex_feat = QgsFeature()
            start_vertex_fields = QgsFields()
            start_vertex_fields.append(QgsField('vertex_id', QVariant.Int, '', 254, 0))
            start_vertex_fields.append(QgsField('cost', QVariant.Double, '', 254, 7))
            start_vertex_fields.append(QgsField('origin_point_id',field_type, '', 254, 7))
            start_vertex_feat.setFields(start_vertex_fields)
            start_vertex_feat['vertex_id'] = current_vertex_id
            start_vertex_feat['cost'] = entry_cost
            start_vertex_feat['origin_point_id'] = current_start_point_id
            pt_m = QgsPoint(self.network.vertex(current_vertex_id).point())
            pt_m.addMValue(entry_cost)
            geom = QgsGeometry(pt_m)
            start_vertex_feat.setGeometry(geom)
            
            iso_pointcloud.update({current_vertex_id: start_vertex_feat})
            
            i = 0
            while i < len(cost):
                #as long as costs at vertex i is greater than iso_distance and there exists an incoming edge (tree[i]!=-1) 
                #consider it as a possible catchment polygon element
                if tree[i] != -1:
                    fromVertexId = self.network.edge(tree[i]).toVertex()
                    real_cost = cost[fromVertexId]+entry_cost
                    #if the costs of the current vertex are lower than the radius, append the vertex id to results.
                    if real_cost <= max_dist:
                        #build feature
                                    
                        feat = QgsFeature()
                        fields = QgsFields()
                        fields.append(QgsField('vertex_id', QVariant.Int, '', 254, 0))
                        fields.append(QgsField('cost', QVariant.Double, '', 254, 7))
                        fields.append(QgsField('origin_point_id',field_type, '', 254, 7))
                        feat.setFields(fields)
                        feat['vertex_id'] = fromVertexId
                        feat['cost'] = real_cost
                        feat['origin_point_id'] = current_start_point_id
                        pt_m = QgsPoint(self.network.vertex(fromVertexId).point())
                        pt_m.addMValue((500-cost[fromVertexId])*2)
                        geom = QgsGeometry(pt_m)
                        feat.setGeometry(geom)
                        
                        if fromVertexId not in iso_pointcloud:
                            #ERROR: FIRST POINT IN POINTCLOUD WILL NEVER BE ADDED
                            iso_pointcloud.update({fromVertexId: feat})
                        if fromVertexId in iso_pointcloud.keys() and iso_pointcloud.get(fromVertexId)['cost'] > real_cost:
                            #if the vertex already exists in the iso_pointcloud and the cost is greater than the existing cost
                            del iso_pointcloud[fromVertexId]
                            #iso_pointcloud.pop(toVertexId)
                            iso_pointcloud.update({fromVertexId: feat})
                        #count up to next vertex
                i = i + 1 
                if (i%10000)==0:
                    self.feedback.pushInfo("[QNEAT3Network][calcIsoPoints] Added {} Nodes to iso pointcloud...".format(i))
                    
        return iso_pointcloud.values() #list of QgsFeature (=QgsFeatureList)
    
    def calcQneatInterpolation(self,iso_pointcloud_featurelist, resolution, interpolation_raster_path):  
        #prepare spatial index
        uri = 'PointM?crs={}&field=vertex_id:int(254)&field=cost:double(254,7)&key=vertex_id&index=yes'.format(self.AnalysisCrs.authid())
        
        mIsoPointcloud = QgsVectorLayer(uri, "mIsoPointcloud_layer", "memory")
        mIsoPointcloud_provider = mIsoPointcloud.dataProvider()
        mIsoPointcloud_provider.addFeatures(iso_pointcloud_featurelist, QgsFeatureSink.FastInsert)
        
        #implement spatial index for lines (closest line, etc...)
        spt_idx = QgsSpatialIndex(mIsoPointcloud.getFeatures(QgsFeatureRequest()), self.feedback)
        
        #prepare numpy coordinate grids
        NoData_value = -9999
        raster_rectangle = mIsoPointcloud.extent()
        
        #top left point
        xmin = raster_rectangle.xMinimum()
        ymin = raster_rectangle.yMinimum()
        xmax = raster_rectangle.xMaximum()
        ymax = raster_rectangle.yMaximum()
        
        cols = int((xmax - xmin) / resolution)
        rows = int((ymax - ymin) / resolution)
        
        output_interpolation_raster = gdal.GetDriverByName('GTiff').Create(interpolation_raster_path, cols, rows, 1, gdal.GDT_Float64 )
        output_interpolation_raster.SetGeoTransform((xmin, resolution, 0, ymax, 0, -resolution))
        
        band = output_interpolation_raster.GetRasterBand(1)
        band.SetNoDataValue(NoData_value)
        
        #initialize zero array with 2 dimensions (according to rows and cols)
        raster_data = zeros(shape=(rows, cols))
        
        #compute raster cell MIDpoints
        x_pos = linspace(xmin+(resolution/2), xmax -(resolution/2), raster_data.shape[1])
        y_pos = linspace(ymax-(resolution/2), ymin + (resolution/2), raster_data.shape[0])
        x_grid, y_grid = meshgrid(x_pos, y_pos) 
        
        self.feedback.pushInfo('[QNEAT3Network][calcQneatInterpolation] Beginning with interpolation')
        total_work = rows * cols
        counter = 0
        
        self.feedback.pushInfo('[QNEAT3Network][calcQneatInterpolation] Total workload: {} cells'.format(total_work))
        self.feedback.setProgress(0)
        for i in range(rows):
            for j in range(cols):
                current_pixel_midpoint = QgsPointXY(x_grid[i,j],y_grid[i,j])
                
                nearest_vertex_fid = spt_idx.nearestNeighbor(current_pixel_midpoint, 1)[0]
                
                nearest_feature = mIsoPointcloud.getFeature(nearest_vertex_fid)
                
                nearest_vertex = self.network.vertex(nearest_feature['vertex_id'])
                
                edges = nearest_vertex.incomingEdges() + nearest_vertex.outgoingEdges()
                
                vertex_found = False
                nearest_counter = 2
                while vertex_found == False:
                    n_nearest_feature_fid = spt_idx.nearestNeighbor(current_pixel_midpoint, nearest_counter)[nearest_counter-1]
                    n_nearest_feature = mIsoPointcloud.getFeature(n_nearest_feature_fid)
                    n_nearest_vertex_id = n_nearest_feature['vertex_id']
                    
                    for edge_id in edges:
                        from_vertex_id = self.network.edge(edge_id).fromVertex()
                        to_vertex_id = self.network.edge(edge_id).toVertex()
                        
                        if n_nearest_vertex_id == from_vertex_id: 
                            vertex_found = True
                            vertex_type = "from_vertex"
                            from_point = n_nearest_feature.geometry().asPoint()
                            from_vertex_cost = n_nearest_feature['cost']
                        if n_nearest_vertex_id == to_vertex_id:
                            vertex_found = True
                            vertex_type = "to_vertex"
                            to_point = n_nearest_feature.geometry().asPoint()
                            to_vertex_cost = n_nearest_feature['cost']
                    
                    nearest_counter = nearest_counter + 1
                    """
                    if nearest_counter == 5:
                        vertex_found = True
                        vertex_type = "end_vertex"
                    """
                
                if vertex_type == "from_vertex":
                    nearest_edge_geometry = QgsGeometry().fromPolylineXY([from_point, nearest_vertex.point()])
                    res = nearest_edge_geometry.closestSegmentWithContext(current_pixel_midpoint)
                    segment_point = res[1] #[0: distance, 1: point, 2: left_of, 3: epsilon for snapping]
                    dist_to_segment = segment_point.distance(current_pixel_midpoint)
                    dist_edge = from_point.distance(segment_point)
                    #self.feedback.pushInfo("dist_to_segment = {}".format(dist_to_segment))
                    #self.feedback.pushInfo("dist_on_edge = {}".format(dist_edge))
                    #self.feedback.pushInfo("cost = {}".format(from_vertex_cost))
                    pixel_cost = from_vertex_cost + dist_edge + dist_to_segment
                    raster_data[i,j] = pixel_cost
                elif vertex_type == "to_vertex":
                    nearest_edge_geometry = QgsGeometry().fromPolylineXY([nearest_vertex.point(), to_point])
                    res = nearest_edge_geometry.closestSegmentWithContext(current_pixel_midpoint)
                    segment_point = res[1] #[0: distance, 1: point, 2: left_of, 3: epsilon for snapping]
                    dist_to_segment = segment_point.distance(current_pixel_midpoint)
                    dist_edge = to_point.distance(segment_point)
                    #self.feedback.pushInfo("dist_to_segment = {}".format(dist_to_segment))
                    #self.feedback.pushInfo("dist_on_edge = {}".format(dist_edge))
                    #self.feedback.pushInfo("cost = {}".format(from_vertex_cost))
                    pixel_cost = to_vertex_cost + dist_edge + dist_to_segment
                    raster_data[i,j] = pixel_cost
                else:
                    pixel_cost = -99999#nearest_feature['cost'] + (nearest_vertex.point().distance(current_pixel_midpoint))
                            
                    
                """
                nearest_feature_pointxy = nearest_feature.geometry().asPoint()
                nearest_feature_cost = nearest_feature['cost']
                
                dist_to_vertex = current_pixel_midpoint.distance(nearest_feature_pointxy)
                #implement time cost
                pixel_cost = dist_to_vertex + nearest_feature_cost
                
                raster_data[i,j] = pixel_cost
                """
                counter = counter+1
                if counter%1000 == 0:
                    self.feedback.pushInfo("[QNEAT3Network][calcQneatInterpolation] Interpolated {} cells...".format(counter))
                self.feedback.setProgress((counter/total_work)*100)
                
                
        band.WriteArray(raster_data)
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromWkt(self.AnalysisCrs.toWkt())
        output_interpolation_raster.SetProjection(outRasterSRS.ExportToWkt())
        band.FlushCache()

        
        
    def calcIsoTinInterpolation(self, iso_point_layer, resolution, interpolation_raster_path):
        if self.AnalysisCrs.isGeographic():
            raise QgsProcessingException('The TIN-Interpolation algorithm in QGIS is designed to work with projected coordinate systems.Please use a projected coordinate system (eg. UTM zones) instead of geographic coordinate systems (eg. WGS84)!')
        
        layer_data = QgsInterpolator.LayerData()
        QgsInterpolator.LayerData
        
        layer_data.source = iso_point_layer #in QGIS2: vectorLayer
        layer_data.valueSource = QgsInterpolator.ValueAttribute
        layer_data.interpolationAttribute =  1 #take second field to get costs
        layer_data.sourceType = QgsInterpolator.SourcePoints

        tin_interpolator = QgsTinInterpolator([layer_data], QgsTinInterpolator.Linear)
        
        rect = iso_point_layer.extent()
        ncol = int((rect.xMaximum() - rect.xMinimum()) / resolution)
        nrows = int((rect.yMaximum() - rect.yMinimum()) / resolution)
        
        writer = QgsGridFileWriter(tin_interpolator, interpolation_raster_path, rect, ncol, nrows)
        writer.writeFile(self.feedback)  # Creating .asc raste
        return QgsRasterLayer(interpolation_raster_path, "temp_qneat3_interpolation_raster")

    def calcIsoContours(self, max_dist, interval, interpolation_raster_path):
        featurelist = []
        
        try:
            import matplotlib.pyplot as plt
        except:
            return featurelist
    
        ds_in = gdal.Open(interpolation_raster_path)
        band_in = ds_in.GetRasterBand(1)
        xsize_in = band_in.XSize
        ysize_in = band_in.YSize
    
        geotransform_in = ds_in.GetGeoTransform()
    
        srs = osr.SpatialReference()
        srs.ImportFromWkt( ds_in.GetProjectionRef() )

        raster_values = band_in.ReadAsArray(0, 0, xsize_in, ysize_in)
        raster_values[raster_values < 0] = max_dist + 1000 #necessary to produce rectangular array from raster
        #nodata values get replaced by the maximum value + 1
        
        x_pos = linspace(geotransform_in[0], geotransform_in[0] + geotransform_in[1] * raster_values.shape[1], raster_values.shape[1])
        y_pos = linspace(geotransform_in[3], geotransform_in[3] + geotransform_in[5] * raster_values.shape[0], raster_values.shape[0])
        x_grid, y_grid = meshgrid(x_pos, y_pos)        
        
        start = interval
        end = interval * ceil(max_dist/interval) +interval
    
        levels = arange(start, end, interval)
        
        fid = 0
        for current_level in nditer(levels):
            self.feedback.pushInfo("[QNEAT3Network][calcIsoContours] Calculating {}-level contours".format(current_level))
            contours = plt.contourf(x_grid, y_grid, raster_values, [0, current_level], antialiased=True)
            
            for collection in contours.collections:
                for contour_paths in collection.get_paths():                    
                    for polygon in contour_paths.to_polygons():
                        x = polygon[:,0]
                        y = polygon[:,1]

                        polylinexy_list = [QgsPointXY(i[0], i[1]) for i in zip(x,y)]
                    
                        feat = QgsFeature()
                        fields = QgsFields()
                        fields.append(QgsField('id', QVariant.Int, '', 254, 0))
                        fields.append(QgsField('cost_level', QVariant.Double, '', 20, 7))
                        feat.setFields(fields)
                        geom = QgsGeometry().fromPolylineXY(polylinexy_list)
                        feat.setGeometry(geom)
                        feat['id'] = fid
                        feat['cost_level'] = float(current_level)
                        featurelist.insert(0, feat)
                        
            fid=fid+1    
        return featurelist
    
    
    def calcIsoPolygons(self, max_dist, interval, interpolation_raster_path):
        featurelist = []
        
        try:
            import matplotlib.pyplot as plt
        except:
            return featurelist
    
        ds_in = gdal.Open(interpolation_raster_path)
        band_in = ds_in.GetRasterBand(1)
        xsize_in = band_in.XSize
        ysize_in = band_in.YSize
    
        geotransform_in = ds_in.GetGeoTransform()
    
        srs = osr.SpatialReference()
        srs.ImportFromWkt( ds_in.GetProjectionRef() )

        raster_values = band_in.ReadAsArray(0, 0, xsize_in, ysize_in)
        raster_values[raster_values < 0] = max_dist + 1000 #necessary to produce rectangular array from raster
        #nodata values get replaced by the maximum value + 1
        
        x_pos = linspace(geotransform_in[0], geotransform_in[0] + geotransform_in[1] * raster_values.shape[1], raster_values.shape[1])
        y_pos = linspace(geotransform_in[3], geotransform_in[3] + geotransform_in[5] * raster_values.shape[0], raster_values.shape[0])
        x_grid, y_grid = meshgrid(x_pos, y_pos)        
        
        start = interval
        end = interval * ceil(max_dist/interval) +interval
    
        levels = arange(start, end, interval)

        fid = 0
        for current_level in nditer(levels):
            self.feedback.pushInfo("[QNEAT3Network][calcIsoPolygons] calculating {}-level contours".format(current_level))
            contours = plt.contourf(x_grid, y_grid, raster_values, [0, current_level], antialiased=True)
        

            
            for collection in contours.collections:
                for contour_path in collection.get_paths(): 
        
                    polygon_list = []
                    
                    for vertex in contour_path.to_polygons():
                        x = vertex[:,0]
                        y = vertex[:,1]

                        polylinexy_list = [QgsPointXY(i[0], i[1]) for i in zip(x,y)]
                        polygon_list.append(polylinexy_list)
                    
                    feat = QgsFeature()
                    fields = QgsFields()
                    fields.append(QgsField('id', QVariant.Int, '', 254, 0))
                    fields.append(QgsField('cost_level', QVariant.Double, '', 20, 7))
                    feat.setFields(fields)
                    geom = QgsGeometry().fromPolygonXY(polygon_list)
                    feat.setGeometry(geom)
                    feat['id'] = fid
                    feat['cost_level'] = float(current_level)
                    

                    featurelist.insert(0, feat)
            fid=fid+1    
        """Maybe move to algorithm"""
        #featurelist = featurelist[::-1] #reverse
        self.feedback.pushInfo("[QNEAT3Network][calcIsoPolygons] number of elements in contour_featurelist: {}".format(len(featurelist)))
        return featurelist
        
class Qneat3AnalysisPoint():
    
    def __init__(self, layer_name, feature, point_id_field_name, network, crs, strategy_int, default_speed, vertex_geom, entry_cost_calculation_method):
        self.layer_name = layer_name
        self.point_feature = feature
        self.point_id = feature[point_id_field_name] 
        self.point_geom = feature.geometry().asPoint()
        self.network_vertex_id = self.getNearestVertexId(network, vertex_geom)
        self.network_vertex = self.getNearestVertex(network, vertex_geom)
        self.crs = crs
        self.strategy = strategy_int
        self.entry_speed = default_speed
        if entry_cost_calculation_method == 0:
            self.entry_cost = self.calcEntryCostEllipsoidal()
        elif entry_cost_calculation_method == 1:
            self.entry_cost = self.calcEntryCostPlanar()
        else:
            self.entry_cost = self.calcEntryCostEllipsoidal()
        
    def calcEntryCostEllipsoidal(self):
        dist_calculator = QgsDistanceArea()
        dist_calculator.setSourceCrs(QgsProject().instance().crs(), QgsProject().instance().transformContext())
        dist_calculator.setEllipsoid(QgsProject().instance().crs().ellipsoidAcronym())
        dist = dist_calculator.measureLine([self.point_geom, self.network_vertex.point()])
        if self.strategy == 0:
            return dist
        else:
            distUnit = self.crs.mapUnits()
            unit_to_meter_factor = QgsUnitTypes.fromUnitToUnitFactor(distUnit, QgsUnitTypes.DistanceMeters)
            return dist/(self.entry_speed*(unit_to_meter_factor * 1000.0 / 3600.0)) #length/(m/s) todo: Make dynamic
    
    def calcEntryCostPlanar(self):
        dist = self.calcEntryLinestring().length()
        if self.strategy == 0:
            return dist
        else:
            distUnit = self.crs.mapUnits()
            unit_to_meter_factor = QgsUnitTypes.fromUnitToUnitFactor(distUnit, QgsUnitTypes.DistanceMeters)
            return dist/(self.entry_speed*(unit_to_meter_factor * 1000.0 / 3600.0)) #length/(m/s) todo: Make dynamic


    def calcEntryLinestring(self):
        return QgsGeometry.fromPolylineXY([self.point_geom, self.network_vertex.point()])
    
    def getNearestVertexId(self, network, vertex_geom):
        return network.findVertex(vertex_geom)
        
    def getNearestVertex(self, network, vertex_geom):
        return network.vertex(self.getNearestVertexId(network, vertex_geom))
    
    def __str__(self):
        return u"Qneat3AnalysisPoint: {} analysis_id: {:30} FROM {:30} TO {:30} network_id: {:d}".format(self.layer_name, self.point_id, self.point_geom.__str__(), self.network_vertex.point().__str__(), self.network_vertex_id)    

# End of Qneat3Framework
#################################################

# network settings
strategy_option = 1   # 0 for distance, 1 for driving time
entry_cost_calc_method = ''
directionFieldNameIndex = -1
forwardValue, backwardValue, bothValue = '', '', ''
defaultDirection = QgsVectorLayerDirector.DirectionBoth  # QgsVectorLayerDirector.DirectionForward/DirectionBackward/DirectionBoth
speedFieldName = 'speed'
defaultSpeed = 3.6 # km/hour
tolerance = 0.0


network_layer_name = "OSSroad2450"
origin_layer_name = "votingStations2450"
destination_layer_name = "populationCentroidWithID2450"
output_filename = "drivingTime.csv"

# grab QGIS objects
network_layer = QgsProject.instance().mapLayersByName(network_layer_name)[0]
origin_layer = QgsProject.instance().mapLayersByName(origin_layer_name)[0]
destination_layer = QgsProject.instance().mapLayersByName(destination_layer_name)[0]

# collect related points
from_coord_list = getListOfPoints(origin_layer)
to_coord_list = getListOfPoints(destination_layer)
merged_coords = from_coord_list + to_coord_list

print("Building network...")
distUnit = network_layer.sourceCrs().mapUnits()
meter_to_unit_factor = QgsUnitTypes.fromUnitToUnitFactor(QgsUnitTypes.DistanceMeters, distUnit)
unit_to_meter_factor = QgsUnitTypes.fromUnitToUnitFactor(distUnit, QgsUnitTypes.DistanceMeters)

graphDirector = QgsVectorLayerDirector(network_layer,
      directionFieldNameIndex,
      forwardValue, backwardValue, bothValue,
      defaultDirection)

# results will be shown in hour unit
speedFieldIndex = network_layer.fields().lookupField( speedFieldName )
if strategy_option == 0:
    strategy = QgsNetworkDistanceStrategy()
else:
    strategy = QgsNetworkSpeedStrategy(speedFieldIndex, defaultSpeed, unit_to_meter_factor * 1000.0 / 3600.0)
multiplier = 3600
graphDirector.addStrategy(strategy)

print("Creating network graph")
graphBuilder = QgsGraphBuilder( network_layer.sourceCrs() )

# This is a compute intensive task and may take some time
print("Building network graph...")
start_time = time.time()
list_tiedPoints = graphDirector.makeGraph(graphBuilder, merged_coords)
network = graphBuilder.graph()
print("total build time: {}".format(time.time() - start_time))

# calculation of entry/exit cost of all points by Qneat3AnalysisPoint
network_crs = network_layer.sourceCrs()
start_time = time.time()
from_analysis_points = [ Qneat3AnalysisPoint(origin_layer_name,
      feature, 'precinctID',  # column name for id
      network, network_crs, strategy_option, defaultSpeed,
      list_tiedPoints[i], entry_cost_calc_method) for i, feature in enumerate(getFeaturesFromQgsIterable( origin_layer ))]
print( "from_analysis_points: {0:d}".format(len(from_analysis_points)) )

from_analysis_points_length = len(from_analysis_points)

to_analysis_points = [Qneat3AnalysisPoint(destination_layer_name,
	  feature, 'Meshcode',    # column name for id
	  network,
	  network_layer.sourceCrs(), strategy_option, defaultSpeed,
	  list_tiedPoints[i + from_analysis_points_length],
	  entry_cost_calc_method) for i, feature in enumerate(getFeaturesFromQgsIterable( destination_layer ))]

print( "to_analysis_points: {0:d}".format(len(to_analysis_points)) )
print( "to/from_analysis_points: {}".format(time.time() - start_time))

outputName = os.path.join(QgsProject.instance().homePath(), output_filename)
csvFile = open(outputName, 'w', newline='')
csv_writer = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
csv_writer.writerow(["origin_id","destination_id","entry_cost", "network_cost", "exit_cost", "total_cost"])

for start_point in from_analysis_points:
    start_time = time.time()

    start_point_index = start_point.network_vertex_id

    # destinations are limited where precinctID of end_point is same as that of start_point
    list_of_end_index = [p.network_vertex_id for p in to_analysis_points if p.point_feature['precinctID'] == start_point.point_feature['precinctID']]
    print( 'loop for start point: {0:d}, num of dest points: {1:d}'.format(start_point_index, len(list_of_end_index)) )

    #  Set starting point, then dijkstra function calculate all points; that's the way they are.
    #  resultTree:  array that represents shortest path tree.
    #    resultTree[ vertexIndex ] == inboundingArcIndex if vertex reachable, 
    #    otherwise resultTree[ vertexIndex ] == -1. 
    #    Note that the startVertexIdx will also have a value of -1 and may need special handling by callers.
    #  resultCost: array of the paths costs
    tree, cost = QgsGraphAnalyzer.dijkstra( network, start_point_index, 0)
    
    for end_index in list_of_end_index:
        find_end_points = list(filter(lambda p: p.network_vertex_id == end_index, to_analysis_points))
        assert len( find_end_points ) >= 1, "Why? probably same distance from nodes :{}".format( end_index )
        end_point = find_end_points[0]
        end_point_index = end_point.network_vertex_id
        network_cost = cost[end_point_index]
        entry_cost = start_point.entry_cost
        exit_cost = end_point.entry_cost
        total_cost = entry_cost + network_cost + exit_cost
        csv_writer.writerow([start_point.point_id, end_point.point_id, entry_cost, network_cost, exit_cost, total_cost])
    print( "loop time: {}".format(time.time() - start_time))
