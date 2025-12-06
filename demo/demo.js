// ==============================================================
// 1. KHAI BÁO & LOAD DỮ LIỆU
// ==============================================================

var ca_mau = ee.FeatureCollection("projects/ee-bonglantrungmuoi/assets/ca_mau");
var forest_area = ee.FeatureCollection("projects/ee-bonglantrungmuoi/assets/forest");
var classification = ee.Image("projects/ee-bonglantrungmuoi/assets/classification");
var table_raw = ee.FeatureCollection("projects/ee-bonglantrungmuoi/assets/4labels");

// List tên band để đổi cho các file cũ
var renameS1 = ['vv', 'vh'];
var renameS2 = ['b4', 'b8', 'b11', 'b12', 'ndvi', 'nbr', 'ndmi'];

// --- LOAD CÁC FILE CŨ (S2 Bands & Indices) ---
var s1_t1 = ee.Image("projects/ee-bonglantrungmuoi/assets/S1_2024_02_04_matched_S2_2024_01_30").rename(renameS1);
var s1_t2 = ee.Image("projects/ee-bonglantrungmuoi/assets/S1_2025_02_22_matched_S2_2025_02_28").rename(renameS1);

var s2_t1 = ee.Image("projects/ee-bonglantrungmuoi/assets/S2_2024_01_30").rename(renameS2);
var s2_t2 = ee.Image("projects/ee-bonglantrungmuoi/assets/S2_2025_02_28").rename(renameS2);

// --- LOAD FILE RGB MỚI (True Color) ---
// GEE thường tự đặt tên band là b1, b2, b3 cho file upload
var rgb_t1 = ee.Image("projects/ee-bonglantrungmuoi/assets/S2_RGB_2024_01_30");
var rgb_t2 = ee.Image("projects/ee-bonglantrungmuoi/assets/S2_RGB_2025_02_28");


// ==============================================================
// 2. CẤU HÌNH PALETTE & VISUALIZATION
// ==============================================================

// Palette màu
var viridis = ['440154', '414487', '2a788e', '22a884', '7ad151', 'fde725'];
var plasma = ['0d0887', '5c01a6', '9c179e', 'ed7953', 'fdb42f', 'f0f921'];
var inferno = ['000004', '320a5e', '781c6d', 'ed6925', 'fbb41a', 'fcffa4'];
var ndviPalette = ['8B0000', 'FF69B4', 'FFFFFF', '90EE90', '006400'];
var ndmiPalette = ['d7191c', 'fdae61', 'ffffbf', 'abdda4', '2b83ba'];
var nbrPalette = ['FFFF00', 'FFA500', 'FF0000'];

// Viz Params
var viz = {
    // Ảnh RGB (True Color)
    // Nếu ảnh trắng xóa, đổi max thành 255. Nếu đen, đổi thành 3000.
    rgb: { min: 0, max: 0.3, gamma: 1.4 },

    // Các Band lẻ S2 (0-1)
    b4: { min: 0, max: 1, palette: viridis },
    b8: { min: 0, max: 1, palette: viridis },
    b11: { min: 0, max: 1, palette: plasma },
    b12: { min: 0, max: 1, palette: plasma },

    // Indices (-1 đến 1)
    ndvi: { min: -1, max: 1, palette: ndviPalette },
    ndmi: { min: -1, max: 1, palette: ndmiPalette },
    nbr: { min: -1, max: 1, palette: nbrPalette },

    // Sentinel-1 (-55 đến 30)
    vv: { min: -55, max: 30, palette: inferno },
    vh: { min: -60, max: 15, palette: inferno },

    // Phân loại
    class: { min: 0, max: 3, palette: ['00734C', 'E60000', 'FFD37F', '00C5FF'] }
};


// ==============================================================
// 3. HIỂN THỊ LÊN BẢN ĐỒ
// ==============================================================

Map.centerObject(forest_area, 11);

// --- NỀN ---
Map.addLayer(ca_mau.style({ color: 'red', fillColor: '00000000' }), {}, 'Ranh giới Cà Mau');
Map.addLayer(forest_area.style({ color: 'orange', fillColor: '00000000', width: 2 }), {}, 'Khu vực nghiên cứu');


// --- THỜI GIAN T1 (2024) ---
Map.addLayer(rgb_t1, viz.rgb, 'T1 S2 | Ảnh RGB (True Color)', false);
Map.addLayer(s2_t1.select('b4'), viz.b4, 'T1 S2 | Band 4 (Red)', false);
Map.addLayer(s2_t1.select('b8'), viz.b8, 'T1 S2 | Band 8 (NIR)', false);
Map.addLayer(s2_t1.select('b11'), viz.b11, 'T1 S2 | Band 11 (SWIR1)', false);
Map.addLayer(s2_t1.select('b12'), viz.b12, 'T1 S2 | Band 12 (SWIR2)', false);
Map.addLayer(s2_t1.select('ndvi'), viz.ndvi, 'T1 S2 | NDVI', false);
Map.addLayer(s2_t1.select('ndmi'), viz.ndmi, 'T1 S2 | NDMI', false);
Map.addLayer(s2_t1.select('nbr'), viz.nbr, 'T1 S2 | NBR', false);
Map.addLayer(s1_t1.select('vv'), viz.vv, 'T1 S1 | VV', false);
Map.addLayer(s1_t1.select('vh'), viz.vh, 'T1 S1 | VH', false);


// --- THỜI GIAN T2 (2025) ---
Map.addLayer(rgb_t2, viz.rgb, 'T2 S2 | Ảnh RGB (True Color)', false);
Map.addLayer(s2_t2.select('b4'), viz.b4, 'T2 S2 | Band 4 (Red)', false);
Map.addLayer(s2_t2.select('b8'), viz.b8, 'T2 S2 | Band 8 (NIR)', false);
Map.addLayer(s2_t2.select('b11'), viz.b11, 'T2 S2 | Band 11 (SWIR1)', false);
Map.addLayer(s2_t2.select('b12'), viz.b12, 'T2 S2 | Band 12 (SWIR2)', false);
Map.addLayer(s2_t2.select('ndvi'), viz.ndvi, 'T2 S2 | NDVI', false);
Map.addLayer(s2_t2.select('ndmi'), viz.ndmi, 'T2 S2 | NDMI', false);
Map.addLayer(s2_t2.select('nbr'), viz.nbr, 'T2 S2 | NBR', false);
Map.addLayer(s1_t2.select('vv'), viz.vv, 'T2 S1 | VV', false);
Map.addLayer(s1_t2.select('vh'), viz.vh, 'T2 S1 | VH', false);


// --- KẾT QUẢ & ĐIỂM ---
Map.addLayer(classification.clip(forest_area), viz.class, '>> KẾT QUẢ PHÂN LOẠI <<', true);

var points = table_raw.map(function (f) {
    var geom = ee.Geometry.Point([f.getNumber('x'), f.getNumber('y')], 'EPSG:32648').transform('EPSG:4326', 1);
    return f.setGeometry(geom);
});
var pointVisFunc = function (f) {
    var color = ee.Algorithms.If(ee.Number(f.get('label')).eq(0), '00734C',
        ee.Algorithms.If(ee.Number(f.get('label')).eq(1), 'E60000',
            ee.Algorithms.If(ee.Number(f.get('label')).eq(2), 'FFD37F', '00C5FF')));
    return f.set('style', { color: color, pointSize: 2, width: 1 });
};
Map.addLayer(points.map(pointVisFunc).style({ styleProperty: 'style' }), {}, 'Dữ liệu mẫu', false);


// ==============================================================
// 4. CHÚ GIẢI (LEGEND)
// ==============================================================
var panel = ui.Panel({ style: { position: 'bottom-left', padding: '8px' } });
panel.add(ui.Label('Chú giải phân loại', { fontWeight: 'bold' }));
var makeRow = function (color, name) {
    return ui.Panel({
        widgets: [
            ui.Label({ style: { backgroundColor: '#' + color, padding: '8px', margin: '0 4px 4px 0', border: '1px solid black' } }),
            ui.Label(name)
        ],
        layout: ui.Panel.Layout.Flow('horizontal')
    });
};
panel.add(makeRow('00734C', '0: Rừng ổn định'));
panel.add(makeRow('E60000', '1: Mất rừng'));
panel.add(makeRow('FFD37F', '2: Phi rừng'));
panel.add(makeRow('00C5FF', '3: Hồi phục rừng'));
Map.add(panel);