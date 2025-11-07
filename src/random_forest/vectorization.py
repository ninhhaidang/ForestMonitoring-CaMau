"""
STEP 8: Vectorization (Optional)
Convert binary classification raster to vector polygons
Apply morphological operations and simplification
"""

import numpy as np
import rasterio
from rasterio import features
import geopandas as gpd
from shapely.geometry import shape, mapping
from shapely.ops import unary_union
from scipy import ndimage
import logging
from typing import List

from common.config import (
    VECTOR_CONFIG, OUTPUT_FILES,
    LOG_CONFIG
)

# Setup logging
logging.basicConfig(**LOG_CONFIG)
logger = logging.getLogger(__name__)


class Vectorizer:
    """Class to convert raster to vector polygons"""

    def __init__(self, config: dict = None):
        """
        Initialize Vectorizer

        Args:
            config: Vectorization configuration (default: from config)
        """
        self.config = config if config is not None else VECTOR_CONFIG
        self.polygons_gdf = None

    def apply_morphological_operations(self, binary_map: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations (erosion + dilation) to smooth binary map

        Args:
            binary_map: Binary classification map (0/1)

        Returns:
            Smoothed binary map
        """
        logger.info("\nApplying morphological operations...")

        erosion_kernel = self.config.get('erosion_kernel', 3)
        dilation_kernel = self.config.get('dilation_kernel', 5)

        # Create structuring elements
        erosion_struct = np.ones((erosion_kernel, erosion_kernel))
        dilation_struct = np.ones((dilation_kernel, dilation_kernel))

        # Apply erosion (remove small noise)
        logger.info(f"  - Erosion (kernel size: {erosion_kernel}x{erosion_kernel})")
        eroded = ndimage.binary_erosion(binary_map, structure=erosion_struct)

        # Apply dilation (restore and smooth boundaries)
        logger.info(f"  - Dilation (kernel size: {dilation_kernel}x{dilation_kernel})")
        smoothed = ndimage.binary_dilation(eroded, structure=dilation_struct)

        # Statistics
        original_pixels = np.sum(binary_map)
        smoothed_pixels = np.sum(smoothed)
        logger.info(f"  ✓ Original deforestation pixels: {original_pixels:,}")
        logger.info(f"  ✓ Smoothed deforestation pixels: {smoothed_pixels:,}")
        logger.info(f"  ✓ Difference: {smoothed_pixels - original_pixels:+,} pixels")

        return smoothed.astype(np.uint8)

    def raster_to_polygons(self, classification_map: np.ndarray,
                          transform, crs,
                          apply_morphology: bool = True) -> gpd.GeoDataFrame:
        """
        Convert binary raster to vector polygons

        Args:
            classification_map: Binary classification map
            transform: Rasterio transform
            crs: Coordinate reference system
            apply_morphology: Apply morphological operations

        Returns:
            GeoDataFrame with polygons
        """
        logger.info("\n" + "="*70)
        logger.info("STEP 8: VECTORIZATION")
        logger.info("="*70)

        # Create binary mask (only deforestation class = 1)
        binary_mask = (classification_map == 1).astype(np.uint8)
        logger.info(f"\nBinary mask created:")
        logger.info(f"  - Deforestation pixels (1): {np.sum(binary_mask):,}")
        logger.info(f"  - No deforestation pixels (0): {np.sum(classification_map == 0):,}")

        # Apply morphological operations if requested
        if apply_morphology:
            binary_mask = self.apply_morphological_operations(binary_mask)

        # Extract shapes
        logger.info("\nExtracting polygons from raster...")
        shapes_generator = features.shapes(
            binary_mask,
            mask=binary_mask == 1,
            transform=transform
        )

        # Convert to list of (geometry, value) tuples
        shapes_list = list(shapes_generator)
        logger.info(f"  ✓ Extracted {len(shapes_list)} raw polygons")

        if len(shapes_list) == 0:
            logger.warning("  ⚠ No polygons extracted!")
            return gpd.GeoDataFrame()

        # Convert to GeoDataFrame
        logger.info("\nCreating GeoDataFrame...")
        geometries = []
        values = []

        for geom_dict, value in shapes_list:
            geometries.append(shape(geom_dict))
            values.append(value)

        gdf = gpd.GeoDataFrame({
            'geometry': geometries,
            'class': values
        }, crs=crs)

        logger.info(f"  ✓ Created GeoDataFrame with {len(gdf)} polygons")

        # Calculate areas
        logger.info("\nCalculating polygon areas...")
        gdf['area_m2'] = gdf.geometry.area
        gdf['area_ha'] = gdf['area_m2'] / 10000  # Convert to hectares

        # Statistics before filtering
        logger.info(f"\nBefore filtering:")
        logger.info(f"  - Total polygons: {len(gdf)}")
        logger.info(f"  - Total area: {gdf['area_ha'].sum():.2f} ha")
        logger.info(f"  - Min area: {gdf['area_m2'].min():.2f} m²")
        logger.info(f"  - Max area: {gdf['area_ha'].max():.2f} ha")
        logger.info(f"  - Mean area: {gdf['area_m2'].mean():.2f} m²")

        # Filter by minimum area
        min_area = self.config.get('min_area', 100)
        logger.info(f"\nFiltering polygons (min area: {min_area} m²)...")
        gdf_filtered = gdf[gdf['area_m2'] >= min_area].copy()

        removed_count = len(gdf) - len(gdf_filtered)
        logger.info(f"  ✓ Removed {removed_count} small polygons")

        # Statistics after filtering
        logger.info(f"\nAfter filtering:")
        logger.info(f"  - Remaining polygons: {len(gdf_filtered)}")
        logger.info(f"  - Total area: {gdf_filtered['area_ha'].sum():.2f} ha")
        logger.info(f"  - Min area: {gdf_filtered['area_m2'].min():.2f} m²")
        logger.info(f"  - Max area: {gdf_filtered['area_ha'].max():.2f} ha")
        logger.info(f"  - Mean area: {gdf_filtered['area_m2'].mean():.2f} m²")

        self.polygons_gdf = gdf_filtered

        return gdf_filtered

    def simplify_geometries(self, gdf: gpd.GeoDataFrame = None) -> gpd.GeoDataFrame:
        """
        Simplify polygon geometries to reduce vertices

        Args:
            gdf: GeoDataFrame to simplify (default: self.polygons_gdf)

        Returns:
            Simplified GeoDataFrame
        """
        if gdf is None:
            gdf = self.polygons_gdf

        if gdf is None or len(gdf) == 0:
            logger.warning("No polygons to simplify")
            return gdf

        logger.info("\nSimplifying geometries...")

        tolerance = self.config.get('simplify_tolerance', 10)
        logger.info(f"  - Tolerance: {tolerance} meters")

        # Count vertices before
        vertices_before = sum([len(geom.exterior.coords) for geom in gdf.geometry if geom.geom_type == 'Polygon'])
        logger.info(f"  - Vertices before: {vertices_before:,}")

        # Simplify
        gdf_simplified = gdf.copy()
        gdf_simplified['geometry'] = gdf_simplified.geometry.simplify(
            tolerance,
            preserve_topology=True
        )

        # Count vertices after
        vertices_after = sum([len(geom.exterior.coords) for geom in gdf_simplified.geometry if geom.geom_type == 'Polygon'])
        logger.info(f"  - Vertices after: {vertices_after:,}")
        logger.info(f"  ✓ Reduced by {vertices_before - vertices_after:,} vertices ({(1 - vertices_after/vertices_before)*100:.1f}%)")

        self.polygons_gdf = gdf_simplified

        return gdf_simplified

    def save_vectors(self, gdf: gpd.GeoDataFrame = None,
                    output_path=None, format='GeoJSON'):
        """
        Save vector polygons to file

        Args:
            gdf: GeoDataFrame to save (default: self.polygons_gdf)
            output_path: Path to save file (default: from config)
            format: Output format ('GeoJSON' or 'Shapefile')
        """
        if gdf is None:
            gdf = self.polygons_gdf

        if gdf is None or len(gdf) == 0:
            logger.warning("No polygons to save")
            return

        if output_path is None:
            if format == 'GeoJSON':
                output_path = OUTPUT_FILES['polygons_geojson']
            else:
                output_path = OUTPUT_FILES['polygons_shapefile']

        logger.info(f"\nSaving vectors to: {output_path}")
        logger.info(f"  - Format: {format}")
        logger.info(f"  - Polygons: {len(gdf)}")

        # Create directory if not exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save based on format
        if format == 'GeoJSON':
            gdf.to_file(output_path, driver='GeoJSON')
        elif format == 'Shapefile':
            gdf.to_file(output_path, driver='ESRI Shapefile')
        else:
            logger.error(f"Unsupported format: {format}")
            return

        logger.info(f"  ✓ Vectors saved successfully")

    def vectorize_and_save(self, classification_map: np.ndarray,
                          transform, crs,
                          apply_morphology: bool = True,
                          apply_simplification: bool = True) -> gpd.GeoDataFrame:
        """
        Complete vectorization pipeline: rasterize, simplify, and save

        Args:
            classification_map: Binary classification map
            transform: Rasterio transform
            crs: Coordinate reference system
            apply_morphology: Apply morphological operations
            apply_simplification: Apply geometry simplification

        Returns:
            Final GeoDataFrame
        """
        # Convert to polygons
        gdf = self.raster_to_polygons(
            classification_map,
            transform,
            crs,
            apply_morphology=apply_morphology
        )

        if len(gdf) == 0:
            logger.warning("No polygons to process")
            return gdf

        # Simplify geometries
        if apply_simplification:
            gdf = self.simplify_geometries(gdf)

        # Save
        output_format = self.config.get('output_format', 'GeoJSON')
        self.save_vectors(gdf, format=output_format)

        logger.info("\n" + "="*70)
        logger.info("✓ VECTORIZATION COMPLETED")
        logger.info("="*70)
        logger.info(f"  - Total polygons: {len(gdf)}")
        logger.info(f"  - Total area: {gdf['area_ha'].sum():.2f} ha")
        logger.info(f"  - Output format: {output_format}")
        logger.info("="*70)

        return gdf

    def get_summary(self) -> dict:
        """
        Get vectorization summary

        Returns:
            Dictionary with summary statistics
        """
        if self.polygons_gdf is None or len(self.polygons_gdf) == 0:
            return {}

        gdf = self.polygons_gdf

        summary = {
            'n_polygons': len(gdf),
            'total_area_m2': float(gdf['area_m2'].sum()),
            'total_area_ha': float(gdf['area_ha'].sum()),
            'min_area_m2': float(gdf['area_m2'].min()),
            'max_area_m2': float(gdf['area_m2'].max()),
            'max_area_ha': float(gdf['area_ha'].max()),
            'mean_area_m2': float(gdf['area_m2'].mean()),
            'median_area_m2': float(gdf['area_m2'].median()),
            'crs': str(gdf.crs)
        }

        return summary


def main():
    """Main function to test vectorization"""
    logger.info("Testing Step 8: Vectorization")

    # Import rasterio to read classification raster
    import rasterio

    # Check if classification raster exists
    if not OUTPUT_FILES['classification_raster'].exists():
        logger.error("Classification raster not found. Please run step 7 first.")
        return None

    # Load classification raster
    logger.info(f"\nLoading classification raster from: {OUTPUT_FILES['classification_raster']}")
    with rasterio.open(OUTPUT_FILES['classification_raster']) as src:
        classification_map = src.read(1)
        transform = src.transform
        crs = src.crs

    logger.info(f"  ✓ Loaded raster")
    logger.info(f"  - Shape: {classification_map.shape}")
    logger.info(f"  - CRS: {crs}")

    # Vectorize
    vectorizer = Vectorizer()
    polygons_gdf = vectorizer.vectorize_and_save(
        classification_map,
        transform,
        crs,
        apply_morphology=True,
        apply_simplification=True
    )

    # Print summary
    summary = vectorizer.get_summary()
    if summary:
        logger.info("\nVectorization Summary:")
        logger.info(f"  - Polygons: {summary['n_polygons']}")
        logger.info(f"  - Total area: {summary['total_area_ha']:.2f} ha")
        logger.info(f"  - Mean area: {summary['mean_area_m2']:.2f} m²")
        logger.info(f"  - Max area: {summary['max_area_ha']:.2f} ha")

    return vectorizer, polygons_gdf


if __name__ == "__main__":
    vectorizer, polygons_gdf = main()
