// ========================================
// SCRIPT TẢI SENTINEL-1 - BANDS VV & VH
// ========================================

// === PARAMS ===
var AOI_ASSET = "projects/ee-bonglantrungmuoi/assets/ca_mau";
var S2_DATE_STR = "2024-01-30";
var MAX_S1_DIFF_DAYS = 7;
var SCALE = 10;

// === AOI & TARGET DATE ===
var AOI = ee.FeatureCollection(AOI_ASSET).geometry();
var targetDate = ee.Date(S2_DATE_STR);

// === SENTINEL-1 COLLECTION ===
var s1Collection = ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(AOI)
    .filterDate(targetDate.advance(-MAX_S1_DIFF_DAYS, 'day'),
        targetDate.advance(MAX_S1_DIFF_DAYS, 'day'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'));

// === THÊM THUỘC TÍNH NGÀY ===
var s1WithDate = s1Collection.map(function (img) {
    var dateStr = img.date().format('YYYY-MM-dd');
    var diff = img.date().difference(targetDate, 'day').abs();
    return img.set({
        'date': dateStr,
        'time_diff_days': diff
    });
});

// === TÌM NGÀY GẦN NHẤT VÀ LẤY TẤT CẢ TILES ===
var s1Exists = s1Collection.size().getInfo() > 0;
var nearestDateStr, timeDiffValue, s1DayCollection;

if (s1Exists) {
    var nearest = s1WithDate.sort('time_diff_days').first();
    nearestDateStr = nearest.get('date').getInfo();
    timeDiffValue = nearest.get('time_diff_days').getInfo();
    s1DayCollection = s1WithDate.filter(ee.Filter.eq('date', nearestDateStr));
} else {
    nearestDateStr = 'NO_S1_FOUND';
    timeDiffValue = 999;
}

// ========================================
// XỬ LÝ: LẤY CẢ 2 BANDS VV & VH
// ========================================
function processS1Collection(col) {
    // Mosaic tất cả tiles
    var mosaic = col.mosaic();
    // Select theo thứ tự VV-VH
    var vv_vh = mosaic.select(['VV', 'VH']);
    return vv_vh.toFloat();
}

var s1_features;
if (s1Exists) {
    s1_features = processS1Collection(s1DayCollection).clip(AOI);
} else {
    s1_features = ee.Image.constant([0, 0]).rename(['VV', 'VH']);
}

// === MATCHING INFO ===
print('');
print('🎯 S1-S2 TEMPORAL MATCHING');
print('Target S2 date:', S2_DATE_STR);
print('✅ Matched S1 date:', nearestDateStr);
print('⏱️  Time difference:', timeDiffValue, 'days');
if (s1Exists) {
    print('🗂️  Tiles mosaicked:', s1DayCollection.size().getInfo());
}
print('');

// === VISUALIZATION ===
Map.centerObject(AOI, 9);
Map.addLayer(AOI, { color: 'red' }, 'AOI Cà Mau', true, 0.5);

// VV band
Map.addLayer(s1_features.select('VV'),
    { min: -25, max: 0, palette: ['black', 'white'] },
    'S1 VV - ' + nearestDateStr, true);

// VH band
Map.addLayer(s1_features.select('VH'),
    { min: -30, max: -5, palette: ['black', 'white'] },
    'S1 VH - ' + nearestDateStr, true);

// === EXPORT ===
var filenameBase = "S1_" + String(nearestDateStr).replace(/-/g, "_") +
    "_matched_S2_" + S2_DATE_STR.replace(/-/g, "_");

Export.image.toDrive({
    image: s1_features,
    description: filenameBase,
    folder: "Sentinel-1",
    fileNamePrefix: filenameBase,
    region: AOI,
    scale: SCALE,
    crs: "EPSG:32648",
    maxPixels: 1e13,
    fileFormat: "GeoTIFF"
});
