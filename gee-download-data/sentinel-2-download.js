// ========================================
// SCRIPT TẢI SENTINEL-2 CHO CHANGE DETECTION
// Chỉ lấy: B4, B8, B11, B12 + NDVI, NBR, NDMI
// Export to Google Drive
// ========================================

// === PARAMS ===
var AOI_ASSET = "projects/ee-bonglantrungmuoi/assets/ca_mau";
var DATE_STR = "2024-01-30";         // YYYY-MM-DD (Thay đổi cho T1/T2)
var CLOUD_THR = 50;                   // Cloud probability threshold
var SCALE = 10;                       // m/px

// === AOI & TIME ===
var AOI = ee.FeatureCollection(AOI_ASSET).geometry();
var date = ee.Date(DATE_STR);
var next = date.advance(1, "day");

// === S2 + Cloud Probability ===
var s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(AOI).filterDate(date, next);
var s2cp = ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
    .filterBounds(AOI).filterDate(date, next);

// Join S2 với cloud prob
var join = ee.Join.saveFirst("prob");
var filt = ee.Filter.equals({ leftField: "system:index", rightField: "system:index" });
var joined = ee.ImageCollection(join.apply(s2, s2cp, filt));

// Mask mây
function maskCloud(im) {
    var p = ee.Image(im.get("prob"));
    var cloud = ee.Algorithms.If(p, ee.Image(p).select("probability"), ee.Image.constant(0));
    return ee.Image(im).updateMask(ee.Image(cloud).lt(CLOUD_THR));
}

// CHỈ LẤY 4 BANDS + TÍNH 3 INDICES CẦN THIẾT
function addIndices(im) {
    // Chỉ lấy B4, B8, B11, B12 và chia cho 10000
    var s = im.select(["B4", "B8", "B11", "B12"]).divide(10000)
        .rename(["B4", "B8", "B11", "B12"]);

    var B4 = s.select("B4");   // Red
    var B8 = s.select("B8");   // NIR
    var B11 = s.select("B11");  // SWIR1
    var B12 = s.select("B12");  // SWIR2

    // 3 chỉ số cần thiết
    var NDVI = B8.subtract(B4).divide(B8.add(B4)).rename("NDVI");
    var NBR = B8.subtract(B12).divide(B8.add(B12)).rename("NBR");
    var NDMI = B8.subtract(B11).divide(B8.add(B11)).rename("NDMI");

    return s.addBands([NDVI, NBR, NDMI]).toFloat();
}

// Ảnh một ngày (mosaic nếu nhiều tile)
var s2_features = joined.map(maskCloud).map(addIndices).mosaic()
    .clip(AOI).set({ date: DATE_STR });

// ===== EXPORT TO GOOGLE DRIVE =====
Export.image.toDrive({
    image: s2_features.select(["B4", "B8", "B11", "B12", "NDVI", "NBR", "NDMI"]),
    description: "S2_" + DATE_STR.replace(/-/g, "_"),
    folder: "Sentinel-2",               // Tên folder trên Drive
    fileNamePrefix: "S2_" + DATE_STR.replace(/-/g, "_"),
    region: AOI,
    scale: SCALE,
    crs: "EPSG:32648",
    maxPixels: 1e13,
    fileFormat: "GeoTIFF"                     // Format file
});
