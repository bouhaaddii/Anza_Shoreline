// Google Earth Engine Script for Shoreline Extraction
// Fully automated multi-temporal processing (2018-2025)

// ============================================
// CONFIGURATION
// ============================================

// Define AOI
var aoi = ee.FeatureCollection("projects/ee-aminebouhadi/assets/tifTransfer");

// Define date ranges for seasons
var seasonRanges = {
  'winter': [1, 3],   // Jan-Mar
  'spring': [4, 6],   // Apr-Jun
  'summer': [7, 9],   // Jul-Sep
  'autumn': [10, 12]  // Oct-Dec
};

// Define years to process
var startYear = 2018;
var endYear = 2025;
var years = ee.List.sequence(startYear, endYear);

// Bands to keep
var bands = ['B2', 'B3', 'B4', 'B8', 'B11'];

// ============================================
// FUNCTIONS
// ============================================

// Function to mask clouds using SCL band
function clearSclMask(image) {
  // SCL is categorical: apply its mask before reflectance resampling.
  // Keep dark-area pixels (class 2); reject no-data, defective, shadow,
  // cloud, cirrus, and snow classes.
  var scl = image.select('SCL');
  return scl.neq(0).and(scl.neq(1)).and(scl.neq(3))
    .and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10)).and(scl.neq(11));
}

// Function to compute NDWI
function addNDWI(image) {
  var ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI');
  return image.addBands(ndwi);
}

// Function to compute MNDWI
function addMNDWI(image) {
  var mndwi = image.normalizedDifference(['B3', 'B11']).rename('MNDWI');
  return image.addBands(mndwi);
}

// Function to preprocess image
function preprocessImage(image) {
  var clear = clearSclMask(image);
  return image.select(bands)
    .updateMask(clear)
    .clip(aoi)
    // Only continuous reflectance bands are bilinearly resampled; SCL codes
    // are never interpolated as if they were continuous measurements.
    .resample('bilinear')
    .reproject({crs: 'EPSG:4326', scale: 10})
    .copyProperties(image, ['system:time_start']);
}

// Function to get season dates
function getSeasonDates(year, season) {
  var months = seasonRanges[season];
  var startMonth = months[0];
  var endMonth = months[1];
  
  var startDate = ee.Date.fromYMD(year, startMonth, 1);
  var endDate = ee.Date.fromYMD(year, endMonth, 1).advance(1, 'month');
  
  return {
    'start': startDate,
    'end': endDate
  };
}

// Function to create seasonal composite
function createSeasonalComposite(year, season) {
  var dates = getSeasonDates(year, season);
  var startDate = dates.start;
  var endDate = dates.end;
  
  print('  Season: ' + season + ' | Start: ' + startDate.format('YYYY-MM-dd').getInfo() + 
        ' | End: ' + endDate.format('YYYY-MM-dd').getInfo());
  
  var collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    .filterDate(startDate, endDate)
    .filterBounds(aoi)
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
    .map(preprocessImage);
  
  // Check if collection has images
  var count = collection.size();
  var countInfo = count.getInfo();
  print('  Images found: ' + countInfo);
  
  if (countInfo > 0) {
    // Create median composite
    var composite = collection.median().clip(aoi);
    
    // Add metadata
    return composite.set({
      'year': year,
      'season': season,
      'system:time_start': startDate.millis(),
      'image_count': count,
      'has_data': 1
    });
  } else {
    // Return empty image
    var emptyImage = ee.Image.constant(0).rename('empty').clip(aoi);
    
    return emptyImage.set({
      'year': year,
      'season': season,
      'system:time_start': startDate.millis(),
      'image_count': count,
      'has_data': 0
    });
  }
}

// Function to add indices to composite
function addIndices(composite) {
  composite = addNDWI(composite);
  composite = addMNDWI(composite);
  return composite.select(bands.concat(['MNDWI']));
}

// Function to export image
function exportImage(image, description) {
  var exportParams = {
    image: image,
    description: description,
    folder: 'Shoreline_Extraction_and_Change',
    scale: 10,
    region: aoi.geometry().bounds(),
    maxPixels: 1e13,
    crs: 'EPSG:4326',
    fileFormat: 'GeoTIFF',
    formatOptions: {
      cloudOptimized: true
    }
  };
  
  Export.image.toDrive(exportParams);
}

// Function to process and export a single year and season
function processAndExport(year, season) {
  print('\nProcessing: ' + year + ' - ' + season);
  
  // Create composite
  var composite = createSeasonalComposite(year, season);
  
  // Get the has_data property
  var hasDataProp = composite.get('has_data');
  var hasDataInfo = hasDataProp.getInfo();
  
  if (hasDataInfo === 1) {
    print('  ✓ Data available, adding indices...');
    
    // Add indices
    var processedImage = addIndices(composite);
    
    // Export multispectral image
    var msImage = processedImage.select(bands);
    exportImage(msImage, 'S2_' + year + '_' + season);
    
    // Export MNDWI raster
    var mndwiImage = processedImage.select('MNDWI');
    exportImage(mndwiImage, 'MNDWI_' + year + '_' + season);
    
    print('  ✓ Export tasks created for ' + year + ' ' + season);
    
    return processedImage;
  } else {
    print('  ✗ No data available for ' + year + ' ' + season);
    return null;
  }
}

// ============================================
// MAIN PROCESSING LOOP
// ============================================

print('Starting Shoreline Extraction Processing');
print('Years: ' + startYear + ' to ' + endYear);
print('AOI: ' + aoi.size().getInfo() + ' features');
print('========================================');

// Create list of all year-season combinations
var yearList = years.getInfo();
var seasonList = Object.keys(seasonRanges);

// Store processed images for visualization
var processedImages = [];
var successfulExports = 0;

// Process each year and season
for (var i = 0; i < yearList.length; i++) {
  var year = yearList[i];
  
  for (var j = 0; j < seasonList.length; j++) {
    var season = seasonList[j];
    
    var processed = processAndExport(year, season);
    if (processed) {
      processedImages.push({
        'year': year,
        'season': season,
        'image': processed
      });
      successfulExports++;
    }
  }
}

// Create separator line
var separator = '========================================';
print('\n' + separator);
print('PROCESSING COMPLETED');
print(separator);
print('\nTotal export tasks created: ' + (successfulExports * 2));
print('Successful periods: ' + successfulExports + ' out of ' + (yearList.length * seasonList.length));
print('Check the "Tasks" tab in the Code Editor');
print('to run all export tasks to Google Drive.');
print('\nExport folder: Shoreline_cnn');
print('Files will be named:');
print('  - S2_YYYY_season.tif (multispectral, 5 bands)');
print('  - MNDWI_YYYY_season.tif (MNDWI index)');
print('\n' + separator);

// ============================================
// VISUALIZATION
// ============================================

// Display AOI
Map.centerObject(aoi, 10);
Map.addLayer(aoi, {color: 'FF0000', fillColor: '00000000'}, 'AOI Boundary');

// Try to visualize one of the processed images if available
if (processedImages.length > 0) {
  var sample = processedImages[0];
  var sampleImage = sample.image;
  
  // Check band names - CORRECTION: Use indexOf instead of includes
  var sampleBandNames = sampleImage.bandNames().getInfo();
  print('\nSample processed image bands: ' + JSON.stringify(sampleBandNames));
  
  // Check if bands exist (not empty)
  if (sampleBandNames && sampleBandNames.length > 0) {
    // Check if it's not an empty image (contains 'empty' band)
    var isEmpty = false;
    for (var b = 0; b < sampleBandNames.length; b++) {
      if (sampleBandNames[b] === 'empty') {
        isEmpty = true;
        break;
      }
    }
    
    if (!isEmpty) {
      // Visualize as RGB
      var rgb = sampleImage.select(['B4', 'B3', 'B2']);
      Map.addLayer(rgb, {
        min: 0,
        max: 3000,
        gamma: 1.4
      }, 'Processed RGB: ' + sample.year + ' ' + sample.season);
      
      // Visualize MNDWI
      var mndwi = sampleImage.select('MNDWI');
      Map.addLayer(mndwi, {
        min: -1,
        max: 1,
        palette: ['0000FF', 'FFFFFF', '00FF00']
      }, 'Processed MNDWI: ' + sample.year + ' ' + sample.season);
      
      print('Sample processed image visualized: ' + sample.year + ' ' + sample.season);
    }
  }
}

// Add a simple legend for MNDWI
var legend = ui.Panel({
  style: {
    position: 'bottom-left',
    padding: '8px 15px',
    backgroundColor: 'white'
  }
});

var legendTitle = ui.Label({
  value: 'MNDWI Values',
  style: {
    fontWeight: 'bold',
    fontSize: '14px',
    margin: '0 0 4px 0',
    padding: '0'
  }
});

legend.add(legendTitle);

// Create color boxes for legend
var colors = ['0000FF', 'FFFFFF', '00FF00'];
var labels = ['Water (>0)', 'Transition (~0)', 'Land (<0)'];

for (var c = 0; c < colors.length; c++) {
  var colorBox = ui.Label({
    style: {
      backgroundColor: '#' + colors[c],
      padding: '8px',
      margin: '0 4px 4px 0',
      border: '1px solid black'
    }
  });
  
  var description = ui.Label({
    value: labels[c],
    style: {
      margin: '0 0 4px 0',
      fontSize: '12px'
    }
  });
  
  var row = ui.Panel({
    widgets: [colorBox, description],
    layout: ui.Panel.Layout.Flow('horizontal')
  });
  
  legend.add(row);
}

Map.add(legend);

print('\nVISUALIZATION:');
print('- Red boundary: AOI');
print('- Color image: Sample RGB composite');
print('- Blue/Green: MNDWI water/land classification');
print('- Check Tasks tab for exports');